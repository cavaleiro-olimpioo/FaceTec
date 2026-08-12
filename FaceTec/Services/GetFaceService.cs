namespace FaceTec.Services;

using OpenCvSharp;
using System;
using System.Collections.Generic;

/// <summary>
/// Serviço responsável pela detecção de rostos utilizando a rede neural YuNet do OpenCV.
/// Otimizado com downscale de imagem para máxima performance.
/// </summary>
public class GetFaceService : IDisposable
{
    private readonly FaceDetectorYN _detector;
    private bool _disposed;
    private EmbeddingFaceService service;
    
    // Fator de escala para reduzir a imagem antes da inferência. 
    // Ex: 0.5f significa processar a imagem com metade da resolução, o que quadruplica a velocidade!
    private readonly float _scaleFactor = 0.5f;

    private bool isTested = false;

    private string isSamePerson = "";

    private float score;
    /// <summary>
    /// Construtor do Serviço de Detecção de Faces.
    /// </summary>
    /// <param name="modelPath">Caminho para o arquivo .onnx do modelo YuNet.</param>
    /// <param name="width">Largura original do frame.</param>
    /// <param name="height">Altura original do frame.</param>
    public GetFaceService(string modelPath, int width, int height)
    {
        // Calcula a nova resolução para o detector (menor, portanto mais rápido)
        int scaledWidth = (int)(width * _scaleFactor);
        int scaledHeight = (int)(height * _scaleFactor);
        
        service = new EmbeddingFaceService();
        
        // Instancia o detector YuNet com o tamanho reduzido
        _detector = FaceDetectorYN.Create(
            model: modelPath,
            config: "",
            inputSize: new Size(scaledWidth, scaledHeight),
            scoreThreshold: 0.8f,   // Confiança mínima para considerar um rosto
            nmsThreshold: 0.3f,     // Threshold para supressão não máxima
            topK: 5000              // Quantidade máxima de rostos detectados
        );
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

        // 1. Otimização: Reduzir a imagem para o tamanho esperado pelo detector
        using var smallFrame = new Mat();
        Cv2.Resize(frame, smallFrame, new Size(frame.Width * _scaleFactor, frame.Height * _scaleFactor));

        // 2. Inferência (IA): Encontrar rostos na imagem pequena
        using var faces = new Mat();
        _detector.Detect(smallFrame, faces);

        int rows = faces.Rows;
        if (rows <= 0)
            return frame; // Nenhum rosto encontrado

        // 3. Pós-processamento: Mapear resultados de volta para a resolução original e desenhar
        float inverseScale = 1.0f / _scaleFactor;

        for (int i = 0; i < rows; i++)
        {
            // Os valores retornados pelo detector [0-3] x,y,w,h e [4-13] landmarks estão na escala reduzida.
            // Precisamos multiplicar pelo inverso da escala para voltar ao tamanho real.
            int x = (int)(faces.At<float>(i, 0) * inverseScale);
            int y = (int)(faces.At<float>(i, 1) * inverseScale);
            int w = (int)(faces.At<float>(i, 2) * inverseScale);
            int h = (int)(faces.At<float>(i, 3) * inverseScale);
            float confidence = faces.At<float>(i, 14);

            // Desenhar a bounding box (caixa) no frame original
            var rect = new Rect(x, y, w, h);
            Cv2.Rectangle(frame, rect, Scalar.Lime, 2, LineTypes.AntiAlias);
            
            
            // Escrever a porcentagem de confiança
            Cv2.PutText(
                frame,
                $"{confidence:0.00}",
                new Point(x, y - 5),
                HersheyFonts.HersheySimplex,
                0.5,
                Scalar.Lime,
                1,
                LineTypes.AntiAlias
            );

            // Detecta se a confidencia é maior que 90, se sim, manda para comparação
            if (!isTested)
            {
                if (confidence > 0.9f)
                {
                    using var faceRecive = new Mat(frame, rect);

                    score = service.CompareFace(faceRecive, "../public/test/face/Face1.jpg");
                    
                    isTested = true;
                    
                    if (score >= 0.4f)
                    {
                        isSamePerson = "sim";
                        
                    }
                    else
                    {
                        isSamePerson = "não";
                    }
                }
            }
            


            


            // Desenhar os 5 landmarks (olho esq, olho dir, nariz, boca esq, boca dir)
            for (int lm = 0; lm < 5; lm++)
            {
                int lx = (int)(faces.At<float>(i, 4 + lm * 2) * inverseScale);
                int ly = (int)(faces.At<float>(i, 4 + lm * 2 + 1) * inverseScale);
                Cv2.Circle(frame, new Point(lx, ly), 2, Scalar.Yellow, -1, LineTypes.AntiAlias);
            }
        }
        Console.WriteLine($"Similaridade: {score:F4}, mesma pessoa? {isSamePerson}");
            
        if (isSamePerson != "")
        {
            isTested = false;
            isSamePerson = "";
        }
        return frame;
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

            /*
            var landmarks = new List<Point>();
            for (int lm = 0; lm < 5; lm++)
            {
                int lx = (int)(faces.At<float>(i, 4 + lm * 2) * inverseScale);
                int ly = (int)(faces.At<float>(i, 4 + lm * 2 + 1) * inverseScale);
                landmarks.Add(new Point(lx, ly));
            }
            */

            result.Add(new DetectedFace
            {
                X = x,
                Y = y,
                Width = w,
                Height = h,
                Confidence = confidence,
                // Landmarks = landmarks
            });
        }

        return result;
    }

    /// <summary>
    /// Libera os recursos não gerenciados (modelo na memória).
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        
        service.Dispose();
        _detector.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// DTO (Data Transfer Object) representando as características de um rosto detectado.
/// </summary>
public class DetectedFace
{
    public int X { get; set; }
    public int Y { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public float Confidence { get; set; }
    public List<Point> Landmarks { get; set; } = new();
}
