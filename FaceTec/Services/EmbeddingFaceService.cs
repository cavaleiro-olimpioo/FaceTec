using OpenCvSharp;
using OpenCvSharp.Extensions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FaceTec.Services;

public class EmbeddingFaceService : IDisposable
{
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
    public float CompareFace(Mat faceRecive, string faceBdPath)
    {
        // Converte Mat -> Bitmap -> ImageSharp
        using var bitmapRecive = BitmapConverter.ToBitmap(faceRecive);
        using var streamRecive = new MemoryStream();
        bitmapRecive.Save(streamRecive, System.Drawing.Imaging.ImageFormat.Png);
        streamRecive.Position = 0;

        using var faceUser = Image.Load<Rgb24>(streamRecive);

        using var faceBd = Image.Load<Rgb24>(faceBdPath);

        var embUser = GetEmbedding(faceUser);
        var embBd = GetEmbedding(faceBd);

        return CosineSimilarity(embUser, embBd);
    }

    /// <summary>
    /// Gera embedding de uma imagem de rosto (já recortada e alinhada).
    /// </summary>
    public float[] GetEmbedding(Image<Rgb24> faceImage)
    {
        // Redimensiona para 112x112 (tamanho esperado pelo MobileFaceNet)
        using var resized = faceImage.Clone(x => x.Resize(112, 112));

        // Prepara tensor [1, 3, 112, 112]
        var tensor = new float[1 * 3 * 112 * 112];
        int idx = 0;

        for (int y = 0; y < 112; y++)
        {
            for (int x = 0; x < 112; x++)
            {
                var pixel = resized[x, y];
                // Normalização típica do MobileFaceNet: (x - 127.5) / 127.5
                tensor[idx++] = (pixel.R - 127.5f) / 127.5f;
                tensor[idx++] = (pixel.G - 127.5f) / 127.5f;
                tensor[idx++] = (pixel.B - 127.5f) / 127.5f;
            }
        }

        var inputTensor = new DenseTensor<float>(tensor, new[] { 1, 3, 112, 112 });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = _session.Run(inputs);
        var output = results[0].AsTensor<float>();

        // Embedding 512-D
        var embedding = new float[512];
        Buffer.BlockCopy(output.ToArray(), 0, embedding, 0, embedding.Length * sizeof(float));

        // Normaliza o embedding (importante para similaridade de cosseno)
        Normalize(embedding);

        return embedding;
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