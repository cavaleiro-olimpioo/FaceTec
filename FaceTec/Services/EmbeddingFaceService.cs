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
/// </remarks>
public class EmbeddingFaceService : IDisposable
{
    private const int InputWidth = 112;
    private const int InputHeight = 112;
    private const int EmbeddingSize = 512;
    private const string InputName = "input";

    private readonly InferenceSession _session;
    private readonly string _modelPath = "../Util/YoloModel/w600k_mbf.onnx";

    public EmbeddingFaceService()
    {
        var sessionOptions = new SessionOptions();
        // Opcional: habilitar GPU se tiver ONNX Runtime com CUDA
        // sessionOptions.AppendExecutionProvider_CUDA(0);

        _session = new InferenceSession(_modelPath, sessionOptions);
    }

    /// <summary>
    /// Compara o rosto recortado (Mat) com um rosto do banco (caminho da imagem).
    /// Retorna similaridade de cosseno (0..1, quanto maior, mais parecido).
    /// </summary>
    /// <param name="receivedFace">Rosto detectado no frame, normalmente em BGR.</param>
    /// <param name="storedFacePath">Caminho da imagem de referência salva no disco.</param>
    /// <returns>Similaridade de cosseno entre os embeddings normalizados.</returns>
    /// <exception cref="ArgumentException">Lançada quando o rosto recebido está vazio.</exception>
    /// <exception cref="FileNotFoundException">Lançada quando a imagem de referência não existe.</exception>
    /// <exception cref="InvalidOperationException">Lançada quando a imagem de referência não pode ser carregada.</exception>
    public float CompareFace(Mat receivedFace, string storedFacePath)
    {
        if (receivedFace.Empty())
            throw new ArgumentException("A imagem do rosto recebido está vazia.", nameof(receivedFace));

        if (!File.Exists(storedFacePath))
            throw new FileNotFoundException("Imagem de referência não encontrada.", storedFacePath);

        using var storedFace = Cv2.ImRead(storedFacePath, ImreadModes.Color);
        if (storedFace.Empty())
            throw new InvalidOperationException($"Não foi possível carregar a imagem de referência: {storedFacePath}");

        var embUser = GetEmbedding(receivedFace);
        var embBd = GetEmbedding(storedFace);

        return CosineSimilarity(embUser, embBd);
    }

    /// <summary>
    /// Gera o embedding de uma imagem de rosto já recortada e, idealmente, alinhada.
    /// </summary>
    /// <param name="faceImage">Imagem do rosto em formato OpenCV Mat.</param>
    /// <returns>Vetor de embedding L2-normalizado.</returns>
    /// <exception cref="ArgumentException">Lançada quando a imagem está vazia.</exception>
    public float[] GetEmbedding(Mat faceImage)
    {
        if (faceImage.Empty())
            throw new ArgumentException("A imagem do rosto está vazia.", nameof(faceImage));

        using var bgrFace = EnsureBgr(faceImage);
        using var resized = new Mat();
        Cv2.Resize(bgrFace, resized, new Size(InputWidth, InputHeight));

        // MobileFaceNet recebe tensor NCHW [1, 3, 112, 112].
        var tensor = new float[1 * 3 * InputHeight * InputWidth];
        int channelSize = InputHeight * InputWidth;

        for (int y = 0; y < InputHeight; y++)
        {
            for (int x = 0; x < InputWidth; x++)
            {
                Vec3b pixel = resized.At<Vec3b>(y, x);
                int offset = y * InputWidth + x;

                // OpenCV trabalha com BGR; o modelo recebe RGB normalizado em [-1, 1].
                tensor[offset] = NormalizePixel(pixel.Item2);
                tensor[channelSize + offset] = NormalizePixel(pixel.Item1);
                tensor[channelSize * 2 + offset] = NormalizePixel(pixel.Item0);
            }
        }

        var inputTensor = new DenseTensor<float>(tensor, new[] { 1, 3, InputHeight, InputWidth });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputName, inputTensor)
        };

        using var results = _session.Run(inputs);
        var output = results[0].AsTensor<float>();

        var embedding = new float[EmbeddingSize];
        Buffer.BlockCopy(output.ToArray(), 0, embedding, 0, embedding.Length * sizeof(float));

        Normalize(embedding);

        return embedding;
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
