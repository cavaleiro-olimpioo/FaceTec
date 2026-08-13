using FaceTec.Repositories;
using FaceTec.Util.dataModel;

namespace FaceTec.Services;

using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;


/// <summary>
/// Serviço responsável pela detecção de rostos utilizando a rede neural YuNet do OpenCV.
/// Otimizado com downscale de imagem para máxima performance.
/// </summary>

public sealed class GetFaceService : IDisposable
{
    private readonly GetStudentData _getStudentData;

    private readonly FaceDetectorYN _detector;
    private readonly string _modelPath;
    private readonly EmbeddingFaceService _embeddingService;
    private readonly FaceAlignmentService _alignmentService;

    private readonly Dictionary<string, float[]> _referenceEmbeddingCache = new();

    // Fator de escala para reduzir a imagem antes da inferência.
    // 0.5f significa processar a imagem com metade da resolução.
    private readonly float _scaleFactor = 0.5f;

    private bool _disposed;
    private bool _isTested;
    private bool _isComparing;
    private string _isSamePerson = string.Empty;
    private float _score;


    /// <summary>
    /// Construtor do serviço de detecção de faces.
    /// </summary>
    /// <param name="modelPath">
    /// Caminho para o arquivo .onnx do modelo YuNet.
    /// </param>
    /// <param name="width">
    /// Largura original do frame.
    /// </param>
    /// <param name="height">
    /// Altura original do frame.
    /// </param>
    /// <param name="connectionString">
    /// String de conexão com o banco de dados.
    /// </param>
    public GetFaceService(
        string modelPath,
        int width,
        int height,
        string connectionString)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException(
                "O caminho do modelo não pode ser vazio.",
                nameof(modelPath));
        }

        if (string.IsNullOrWhiteSpace(connectionString))
        {
            throw new ArgumentException(
                "A connection string não pode ser vazia.",
                nameof(connectionString));
        }

        _modelPath = modelPath;

        // Cria o serviço responsável pelas consultas ao banco.
        _getStudentData = new GetStudentData(connectionString);

        // Calcula a nova resolução para o detector.
        int scaledWidth = (int)(width * _scaleFactor);
        int scaledHeight = (int)(height * _scaleFactor);

        if (scaledWidth <= 0 || scaledHeight <= 0)
        {
            throw new ArgumentException(
                "A largura e a altura precisam ser maiores que zero.");
        }

        _embeddingService = new EmbeddingFaceService();
        _alignmentService = new FaceAlignmentService();

        // Instancia o detector YuNet com o tamanho reduzido.
        _detector = FaceDetectorYN.Create(
            model: modelPath,
            config: string.Empty,
            inputSize: new Size(scaledWidth, scaledHeight),
            scoreThreshold: 0.8f,
            nmsThreshold: 0.3f,
            topK: 5000);
    }
    /// <summary>
    /// Detecta rostos de forma otimizada utilizando uma versão reduzida da imagem.
    /// Após a detecção, desenha as caixas delimitadoras e landmarks diretamente no frame original (alta resolução).
    /// </summary>
    /// <param name="frame">O frame original (alta resolução) capturado da câmera/vídeo.</param>
    /// <returns>O mesmo frame com os rostos desenhados.</returns>
    public Mat DrawFaces(Mat frame)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GetFaceService));

        var detectedFaces = DetectFaces(frame);
        if (detectedFaces.Count <= 0)
            return frame; // Nenhum rosto encontrado

        foreach (var detectedFace in detectedFaces)
        {
            var rect = new Rect(detectedFace.X, detectedFace.Y, detectedFace.Width, detectedFace.Height);
            if (rect.Width <= 0 || rect.Height <= 0)
                continue;

            // Desenhar a bounding box (caixa) no frame original
            Cv2.Rectangle(frame, rect, Scalar.Lime, 2, LineTypes.AntiAlias);
            
            
            // Escrever a porcentagem de confiança
            Cv2.PutText(
                frame,
                $"{detectedFace.Confidence:0.00}",
                new Point(rect.X, rect.Y - 5),
                HersheyFonts.HersheySimplex,
                0.5,
                Scalar.Lime,
                1,
                LineTypes.AntiAlias
            );

            // Detecta se a confidencia é maior que 90, se sim, manda para comparação
            if (!_isTested)
            {
                if (!_isComparing)
                {
                    CompareFaces(frame, detectedFace);
                }
            }
            
            if (_isSamePerson.Equals("não"))
            {
                Console.WriteLine("Erro aluno não encontrado");
            }


            if (_isSamePerson != "")
            {
                _isSamePerson = "";
                _isTested = false;
            }

            
            
            
            // Desenhar os 5 landmarks (olho esq, olho dir, nariz, boca esq, boca dir)
            foreach (var landmark in detectedFace.Landmarks)
            {
                Cv2.Circle(
                    frame,
                    new Point((int)landmark.X, (int)landmark.Y),
                    2,
                    Scalar.Yellow,
                    -1,
                    LineTypes.AntiAlias
                );
            }
        }
            
        
        return frame;
    }

    private static Rect ClampRectToFrame(Rect rect, Mat frame)
    {
        int x = Math.Clamp(rect.X, 0, frame.Width);
        int y = Math.Clamp(rect.Y, 0, frame.Height);
        int right = Math.Clamp(rect.X + rect.Width, 0, frame.Width);
        int bottom = Math.Clamp(rect.Y + rect.Height, 0, frame.Height);

        return new Rect(x, y, right - x, bottom - y);
    }

    /// <summary>
    /// Carrega, detecta, alinha e gera o embedding de uma imagem de referência.
    /// </summary>
    /// <remarks>
    /// A imagem salva no banco precisa passar pelo mesmo pipeline do frame ao vivo.
    /// Comparar um rosto ao vivo alinhado contra uma referência crua reintroduz variação
    /// de rotação, escala e posição, anulando o ganho do alinhamento ArcFace.
    /// O resultado é cacheado por caminho absoluto para não repetir inferências a cada frame.
    /// </remarks>
    /// <param name="referenceFacePath">Caminho da imagem de referência no disco.</param>
    /// <returns>Embedding normalizado da referência alinhada.</returns>
    /// <exception cref="FileNotFoundException">Lançada quando a imagem não existe.</exception>
    /// <exception cref="InvalidOperationException">Lançada quando a imagem não pode ser carregada ou não possui rosto detectável.</exception>
    private float[] GetReferenceEmbedding(string referenceFacePath)
    {
        string fullPath = Path.GetFullPath(referenceFacePath);
        if (_referenceEmbeddingCache.TryGetValue(fullPath, out var cachedEmbedding))
            return cachedEmbedding;

        if (!File.Exists(fullPath))
            throw new FileNotFoundException("Imagem de referência não encontrada.", fullPath);

        using var referenceImage = Cv2.ImRead(fullPath, ImreadModes.Color);
        if (referenceImage.Empty())
            throw new InvalidOperationException($"Não foi possível carregar a imagem de referência: {fullPath}");

        var referenceFace = DetectFacesInImage(referenceImage)
            .OrderByDescending(face => face.Confidence)
            .FirstOrDefault();

        if (referenceFace is null)
            throw new InvalidOperationException($"Nenhum rosto foi detectado na imagem de referência: {fullPath}");

        using var alignedReferenceFace = _alignmentService.Align(referenceImage, referenceFace.Landmarks);
        var embedding = _embeddingService.GetEmbedding(alignedReferenceFace);
        _referenceEmbeddingCache[fullPath] = embedding;

        return embedding;
    }

    /// <summary>
    /// Versão que retorna apenas a lista de dados brutos dos rostos encontrados (sem desenhar).
    /// Útil caso queria utilizar as coordenadas para salvar no banco ou recortar os rostos.
    /// </summary>
    /// <param name="frame">Frame original de entrada.</param>
    /// <returns>Lista de objetos DetectedFace contendo as coordenadas em alta resolução.</returns>
    public List<DetectedFace> DetectFaces(Mat frame)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GetFaceService));

        // Mesma otimização de resize
        using var smallFrame = new Mat();
        Cv2.Resize(frame, smallFrame, new Size(frame.Width * _scaleFactor, frame.Height * _scaleFactor));

        using var faces = new Mat();
        _detector.Detect(smallFrame, faces);

        var result = new List<DetectedFace>();
        int rows = faces.Rows;
        
        float inverseScale = 1.0f / _scaleFactor;

        for (int i = 0; i < rows; i++)
        {
            int x = (int)(faces.At<float>(i, 0) * inverseScale);
            int y = (int)(faces.At<float>(i, 1) * inverseScale);
            int w = (int)(faces.At<float>(i, 2) * inverseScale);
            int h = (int)(faces.At<float>(i, 3) * inverseScale);
            float confidence = faces.At<float>(i, 14);
            var rect = ClampRectToFrame(new Rect(x, y, w, h), frame);
            if (rect.Width <= 0 || rect.Height <= 0)
                continue;

            var landmarks = ExtractLandmarks(faces, i, inverseScale);

            result.Add(new DetectedFace
            {
                X = rect.X,
                Y = rect.Y,
                Width = rect.Width,
                Height = rect.Height,
                Confidence = confidence,
                Landmarks = landmarks
            });
        }

        return result;
    }

    /// <summary>
    /// Detecta rostos em uma imagem de referência usando o tamanho original da imagem.
    /// </summary>
    /// <remarks>
    /// O detector principal é configurado para o stream ao vivo reduzido. Como o pacote atual
    /// do OpenCvSharp não expõe SetInputSize para reutilizar esse detector com outro tamanho,
    /// criamos um detector temporário apenas quando uma referência ainda não está no cache.
    /// Isso mantém o custo fora do caminho crítico por frame.
    /// </remarks>
    /// <param name="image">Imagem de referência carregada do disco.</param>
    /// <returns>Lista de rostos detectados com landmarks na escala original da imagem.</returns>
    private List<DetectedFace> DetectFacesInImage(Mat image)
    {
        using var detector = FaceDetectorYN.Create(
            model: _modelPath,
            config: "",
            inputSize: new Size(image.Width, image.Height),
            scoreThreshold: 0.8f,
            nmsThreshold: 0.3f,
            topK: 5000
        );

        using var faces = new Mat();
        detector.Detect(image, faces);

        var result = new List<DetectedFace>();
        int rows = faces.Rows;
        for (int i = 0; i < rows; i++)
        {
            int x = (int)faces.At<float>(i, 0);
            int y = (int)faces.At<float>(i, 1);
            int w = (int)faces.At<float>(i, 2);
            int h = (int)faces.At<float>(i, 3);
            var rect = ClampRectToFrame(new Rect(x, y, w, h), image);
            if (rect.Width <= 0 || rect.Height <= 0)
                continue;

            result.Add(new DetectedFace
            {
                X = rect.X,
                Y = rect.Y,
                Width = rect.Width,
                Height = rect.Height,
                Confidence = faces.At<float>(i, 14),
                Landmarks = ExtractLandmarks(faces, i, 1.0f)
            });
        }

        return result;
    }

    private static List<Point2f> ExtractLandmarks(Mat faces, int row, float scale)
    {
        var landmarks = new List<Point2f>(5);
        for (int lm = 0; lm < 5; lm++)
        {
            float lx = faces.At<float>(row, 4 + lm * 2) * scale;
            float ly = faces.At<float>(row, 4 + lm * 2 + 1) * scale;
            landmarks.Add(new Point2f(lx, ly));
        }

        return landmarks;
    }

    /// <summary>
    /// Libera os recursos não gerenciados (modelo na memória).
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        
        _embeddingService.Dispose();
        _detector.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Compara os rostos e entrega o real
    /// </summary>
    private async void CompareFaces(Mat frame, DetectedFace detectedFace)
    {
        if (detectedFace.Confidence > 0.9f)
        {
            if (_isComparing)
                return;
            
            _isComparing = true;
            
            using var frameCopy = frame.Clone();
            
            int count = 1;
            while (true)
            {
                Console.WriteLine($"Antes do GetStudentPictureAsync: {count}");

                byte[]? aluno = await _getStudentData.GetStudentPictureAsync(count);

                
                Console.WriteLine(
                    aluno == null
                        ? "GetStudentPictureAsync RETORNOU NULL"
                        : $"GetStudentPictureAsync RETORNOU {aluno.Length} BYTES");
                
                if (aluno == null)
                {
                    Console.WriteLine("Detectou aluno null");
                    _isSamePerson = "não";
                    break;
                }
                else
                {
                    Console.WriteLine("Se chegou aqui pq não ta funcionando kkkkkk");
                    await File.WriteAllBytesAsync("student.jpg", aluno);
                    
                    try
                    {
                        using var alignedReceivedFace = _alignmentService.Align(frameCopy, detectedFace.Landmarks);
                        var receivedEmbedding = _embeddingService.GetEmbedding(alignedReceivedFace);
                        var referenceEmbedding = GetReferenceEmbedding("student.jpg");

                        _score = _embeddingService.CompareEmbeddings(receivedEmbedding, referenceEmbedding);
                        if (_score >= 0.4f)
                        {
                            StudentModel? DataAluno = await _getStudentData.GetStudentDataByIdAsync(count);
                            Console.WriteLine($"Bem vindo {DataAluno.nome}");
                            _isSamePerson = "sim";
                            break;
                        }

                        count++;

                        _isTested = true;
                        
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine($"Erro ao comparar rosto: {ex.Message}");
                        
                        count++;
                    }
                    finally
                    {
                        _isComparing = false;
                    }
                    
                }
                

            }
        }
        
    }
}

/// <summary>
/// DTO (Data Transfer Object) representando as características de um rosto detectado.
/// </summary>
public class DetectedFace
{
    /// <summary>
    /// Coordenada X da bounding box do rosto na imagem original.
    /// </summary>
    public int X { get; set; }

    /// <summary>
    /// Coordenada Y da bounding box do rosto na imagem original.
    /// </summary>
    public int Y { get; set; }

    /// <summary>
    /// Largura da bounding box do rosto.
    /// </summary>
    public int Width { get; set; }

    /// <summary>
    /// Altura da bounding box do rosto.
    /// </summary>
    public int Height { get; set; }

    /// <summary>
    /// Confiança retornada pelo YuNet para a detecção.
    /// </summary>
    public float Confidence { get; set; }

    /// <summary>
    /// Landmarks na ordem ArcFace: olho esquerdo, olho direito, nariz, boca esquerda e boca direita.
    /// </summary>
    /// <remarks>
    /// Esses pontos são preservados na escala da imagem original para que o alinhamento seja aplicado
    /// no frame cheio, não na versão reduzida usada apenas para acelerar a detecção.
    /// </remarks>
    public List<Point2f> Landmarks { get; set; } = new();
}


