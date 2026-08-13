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

    // ---- CORRIGIDO: cache indexado por ID do aluno, não por caminho de arquivo ----
    // Antes era Dictionary<string, float[]> chaveado em "student.jpg" (caminho fixo).
    // Como todo aluno era gravado no MESMO arquivo antes de calcular o embedding,
    // o cache só era preenchido uma vez (no primeiro aluno comparado desde que o
    // programa iniciou) e todo mundo depois disso era comparado contra aquele mesmo
    // embedding — por isso o sistema "travava" sempre na primeira pessoa testada.
    private readonly Dictionary<int, float[]> _referenceEmbeddingCache = new();

    // Fator de escala para reduzir a imagem antes da inferência.
    // 0.5f significa processar a imagem com metade da resolução.
    private readonly float _scaleFactor = 0.5f;

    private bool _disposed;
    private bool _isTested;
    private bool _isComparing;
    private string _isSamePerson = string.Empty;
    private float _score;

    private DateTime _nextVerification;

    // ---- Mini interface: estado de reconhecimento exposto pro Program.cs ler e desenhar/tocar o beep ----
    // O Program.cs assina OnRecognized UMA ÚNICA VEZ, fora do loop de frames.
    public event Action<string>? OnRecognized;
    public string? RecognizedName { get; private set; }
    public Mat? RecognizedPhoto { get; private set; }
    private DateTime _recognizedUntil;
    public bool IsRecognitionActive => DateTime.Now < _recognizedUntil;

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
    /// <remarks>
    /// Este método só desenha caixas/landmarks e dispara a comparação. Ele não sabe nada sobre
    /// janela, loop de captura ou overlay de "Bem vindo" — isso é responsabilidade do Program.cs,
    /// que lê RecognizedName/RecognizedPhoto/IsRecognitionActive.
    /// </remarks>
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
                    if (DateTime.Now >= _nextVerification)
                    {
                        CompareFaces(frame, detectedFace);
                    }
                    
                }
            }
            
            if (_isSamePerson.Equals("não"))
            {
                Console.WriteLine("Erro aluno não encontrado");
                _nextVerification = DateTime.Now.AddSeconds(5);
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
    /// <param name="image">Imagem de referência decodificada em memória.</param>
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

        RecognizedPhoto?.Dispose();

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
                byte[]? aluno = await _getStudentData.GetStudentPictureAsync(count);

                if (aluno == null)
                {
                    _isSamePerson = "não";
                    break;
                }
                else
                {
                    try
                    {
                        // ---- CORRIGIDO: embedding de referência cacheado por ID do aluno (`count`) ----
                        // Antes: gravava sempre em "student.jpg" e o cache (por caminho de arquivo)
                        // só era populado na primeira vez — todo mundo depois comparava contra o
                        // embedding da primeira pessoa testada. Agora cada aluno decodifica direto
                        // dos bytes vindos do banco (sem tocar disco) e cacheia pelo próprio id.
                        if (!_referenceEmbeddingCache.TryGetValue(count, out var referenceEmbedding))
                        {
                            using var referenceImage = Cv2.ImDecode(aluno, ImreadModes.Color);
                            if (referenceImage.Empty())
                            {
                                Console.Error.WriteLine($"Foto do aluno {count} inválida/corrompida, pulando.");
                                count++;
                                continue;
                            }

                            var referenceFace = DetectFacesInImage(referenceImage)
                                .OrderByDescending(face => face.Confidence)
                                .FirstOrDefault();

                            if (referenceFace is null)
                            {
                                Console.Error.WriteLine($"Nenhum rosto detectado na foto do aluno {count}, pulando.");
                                count++;
                                continue;
                            }

                            using var alignedReferenceFace = _alignmentService.Align(referenceImage, referenceFace.Landmarks);
                            referenceEmbedding = _embeddingService.GetEmbedding(alignedReferenceFace);
                            _referenceEmbeddingCache[count] = referenceEmbedding;
                        }

                        using var alignedReceivedFace = _alignmentService.Align(frameCopy, detectedFace.Landmarks);
                        var receivedEmbedding = _embeddingService.GetEmbedding(alignedReceivedFace);

                        _score = _embeddingService.CompareEmbeddings(receivedEmbedding, referenceEmbedding);
                        if (_score >= 0.4f)
                        {
                            StudentModel? dataAluno = await _getStudentData.GetStudentDataByIdAsync(count);

                            // ---- Mini interface: guarda estado de reconhecimento pro Program.cs desenhar/tocar o beep ----
                            RecognizedPhoto?.Dispose();
                            RecognizedPhoto = Cv2.ImDecode(aluno, ImreadModes.Color); // decodifica da memória, sem I/O extra
                            RecognizedName = dataAluno.nome;
                            _recognizedUntil = DateTime.Now.AddSeconds(5);

                            OnRecognized?.Invoke(dataAluno.nome);

                            _nextVerification = DateTime.Now.AddSeconds(5);
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