using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FaceTec.Services;

/// <summary>
/// Serviço responsável por gerar embeddings faciais com o modelo MobileFaceNet
/// e comparar dois rostos por similaridade de cosseno.
/// </summary>
/// <remarks>
/// O pré-processamento usa somente OpenCV para evitar a dependência do ImageSharp.
/// A versão 4.x do ImageSharp exige uma licença Six Labors no build, então ela foi
/// removida do projeto. Se o pacote voltar a ser adicionado futuramente, será preciso
/// configurar uma licença válida conforme os termos da Six Labors.
/// O arquivo <c>Util/YoloModel/w600k_mbf.onnx</c> é copiado para a pasta de saída
/// pelo projeto e carregado a partir de <see cref="AppContext.BaseDirectory"/>.
/// O nome da entrada do tensor é lido do metadata do ONNX para evitar acoplamento
/// com nomes específicos exportados pelo modelo.
/// </remarks>
public class EmbeddingFaceService : IDisposable
{
    private const int EmbeddingSize = 512;
    private const string EmbeddingModelFileName = "w600k_mbf.onnx";

    private readonly InferenceSession _session;
    private readonly string _inputName;

    public EmbeddingFaceService()
    {
        var sessionOptions = new SessionOptions();
        // Opcional: habilitar GPU se tiver ONNX Runtime com CUDA
        // sessionOptions.AppendExecutionProvider_CUDA(0);

        string modelPath = Path.Combine(
            AppContext.BaseDirectory,
            "Util",
            "YoloModel",
            EmbeddingModelFileName
        );

        if (!File.Exists(modelPath))
            throw new FileNotFoundException("Modelo de embedding facial não encontrado.", modelPath);

        _session = new InferenceSession(modelPath, sessionOptions);
        _inputName = _session.InputMetadata.Keys.First();
    }

    /// <summary>
    /// Gera o embedding de uma imagem de rosto já alinhada no padrão ArcFace.
    /// </summary>
    /// <remarks>
    /// O modelo w600k_mbf espera rostos alinhados em 112x112. O método ainda redimensiona
    /// a imagem por tolerância operacional, mas a correção geométrica deve acontecer antes,
    /// no serviço de alinhamento, para que a comparação por cosseno tenha poder discriminativo.
    /// </remarks>
    /// <param name="alignedFace">Rosto alinhado em formato OpenCV Mat.</param>
    /// <returns>Vetor de embedding L2-normalizado.</returns>
    /// <exception cref="ArgumentException">Lançada quando a imagem está vazia.</exception>
    public float[] GetEmbedding(Mat alignedFace)
    {
        if (alignedFace.Empty())
            throw new ArgumentException("A imagem do rosto alinhado está vazia.", nameof(alignedFace));

        using var bgrFace = EnsureBgr(alignedFace);
        using var resized = new Mat();
        Cv2.Resize(bgrFace, resized, new Size(FaceAlignmentService.AlignedWidth, FaceAlignmentService.AlignedHeight));

        // MobileFaceNet recebe tensor NCHW [1, 3, 112, 112].
        var tensor = new float[1 * 3 * FaceAlignmentService.AlignedHeight * FaceAlignmentService.AlignedWidth];
        int channelSize = FaceAlignmentService.AlignedHeight * FaceAlignmentService.AlignedWidth;

        for (int y = 0; y < FaceAlignmentService.AlignedHeight; y++)
        {
            for (int x = 0; x < FaceAlignmentService.AlignedWidth; x++)
            {
                Vec3b pixel = resized.At<Vec3b>(y, x);
                int offset = y * FaceAlignmentService.AlignedWidth + x;

                // OpenCV trabalha com BGR; o modelo recebe RGB normalizado em [-1, 1].
                tensor[offset] = NormalizePixel(pixel.Item2);
                tensor[channelSize + offset] = NormalizePixel(pixel.Item1);
                tensor[channelSize * 2 + offset] = NormalizePixel(pixel.Item0);
            }
        }

        var inputTensor = new DenseTensor<float>(
            tensor,
            new[] { 1, 3, FaceAlignmentService.AlignedHeight, FaceAlignmentService.AlignedWidth }
        );
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        using var results = _session.Run(inputs);
        var output = results[0].AsTensor<float>();

        var embedding = new float[EmbeddingSize];
        Buffer.BlockCopy(output.ToArray(), 0, embedding, 0, embedding.Length * sizeof(float));

        Normalize(embedding);

        return embedding;
    }

    /// <summary>
    /// Compara dois embeddings já normalizados por similaridade de cosseno.
    /// </summary>
    /// <remarks>
    /// Separar a comparação do carregamento da imagem permite cachear embeddings de referência.
    /// Assim, o rosto salvo no banco passa uma única vez pelo pipeline detecção-alinhamento-embedding
    /// e não precisa ser recalculado a cada frame da câmera.
    /// </remarks>
    /// <param name="receivedEmbedding">Embedding do rosto capturado ao vivo.</param>
    /// <param name="referenceEmbedding">Embedding do rosto de referência.</param>
    /// <returns>Similaridade de cosseno entre -1 e 1.</returns>
    /// <exception cref="ArgumentException">Lançada quando os vetores têm tamanhos diferentes.</exception>
    public float CompareEmbeddings(float[] receivedEmbedding, float[] referenceEmbedding)
    {
        if (receivedEmbedding.Length != referenceEmbedding.Length)
            throw new ArgumentException("Os embeddings comparados precisam ter o mesmo tamanho.");

        return CosineSimilarity(receivedEmbedding, referenceEmbedding);
    }

    private static float NormalizePixel(byte value)
    {
        return (value - 127.5f) / 127.5f;
    }

    private static Mat EnsureBgr(Mat image)
    {
        if (image.Channels() == 3)
            return image.Clone();

        var converted = new Mat();
        if (image.Channels() == 1)
        {
            Cv2.CvtColor(image, converted, ColorConversionCodes.GRAY2BGR);
            return converted;
        }

        if (image.Channels() == 4)
        {
            Cv2.CvtColor(image, converted, ColorConversionCodes.BGRA2BGR);
            return converted;
        }

        converted.Dispose();
        throw new NotSupportedException($"Formato de imagem com {image.Channels()} canais não suportado.");
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        // Se já estiverem normalizados, é só o produto escalar
        float dot = 0f;
        for (int i = 0; i < a.Length; i++)
            dot += a[i] * b[i];
        return dot; // valor entre -1 e 1, normalmente 0..1
    }

    private static void Normalize(float[] v)
    {
        float norm = 0f;
        for (int i = 0; i < v.Length; i++)
            norm += v[i] * v[i];
        norm = (float)Math.Sqrt(norm);
        if (norm <= 0f) return;
        for (int i = 0; i < v.Length; i++)
            v[i] /= norm;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
