namespace FaceTec.Services;

using OpenCvSharp;
using System;
using System.Collections.Generic;

public class GetFaceService : IDisposable
{
    private readonly FaceDetectorYN _detector;
    private bool _disposed;

    /// <param name="modelPath">Caminho para o arquivo .onnx (YuNet)</param>
    /// <param name="width">Largura do frame (mesmo valor usado no ffmpeg)</param>
    /// <param name="height">Altura do frame (mesmo valor usado no ffmpeg)</param>
    public GetFaceService(string modelPath, int width, int height)
    {
        // Cria o detector YuNet com os mesmos width/height do vídeo
        _detector = FaceDetectorYN.Create(
            model: modelPath,
            config: "",
            inputSize: new Size(width, height),
            scoreThreshold: 0.9f,   // confiança mínima
            nmsThreshold: 0.3f,     // NMS threshold
            topK: 5000              // máximo de faces
        );
    }

    /// <summary>
    /// Processa um frame e desenha as faces detectadas na própria imagem.
    /// Retorna o Mat com as bounding boxes e landmarks.
    /// </summary>
    public Mat DrawFaces(Mat frame)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GetFaceService));

        // Garantir que o detector está usando o tamanho do frame atual
        // O OpenCvSharp4 atual não expõe SetInputSize, então assumimos que o frame tem o tamanho passado no construtor.

        using var faces = new Mat();
        _detector.Detect(frame, faces);

        int rows = faces.Rows;
        if (rows <= 0)
            return frame;

        for (int i = 0; i < rows; i++)
        {
            // Cada linha tem 15 valores:
            // [0-3]  = x, y, w, h
            // [4-13] = 5 landmarks (x,y)
            // [14]   = confidence
            int x = (int)faces.At<float>(i, 0);
            int y = (int)faces.At<float>(i, 1);
            int w = (int)faces.At<float>(i, 2);
            int h = (int)faces.At<float>(i, 3);
            float confidence = faces.At<float>(i, 14);

            // Desenhar bounding box
            var rect = new Rect(x, y, w, h);
            Cv2.Rectangle(frame, rect, Scalar.Lime, 2, LineTypes.AntiAlias);

            // Escrever confiança
            Cv2.PutText(
                frame,
                $"{confidence:0.00}",
                new Point(x, y - 5),
                HersheyFonts.HersheySimplex,
                0.5,
                Scalar.Red,
                1,
                LineTypes.AntiAlias
            );

            // Desenhar 5 landmarks
            for (int lm = 0; lm < 5; lm++)
            {
                int lx = (int)faces.At<float>(i, 4 + lm * 2);
                int ly = (int)faces.At<float>(i, 4 + lm * 2 + 1);
                Cv2.Circle(frame, new Point(lx, ly), 2, Scalar.Yellow, -1, LineTypes.AntiAlias);
            }
        }

        return frame;
    }

    /// <summary>
    /// Versão que retorna os dados das faces, caso você queira usar depois.
    /// </summary>
    public List<DetectedFace> DetectFaces(Mat frame)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GetFaceService));

        // O tamanho já foi definido no construtor.

        using var faces = new Mat();
        _detector.Detect(frame, faces);

        var result = new List<DetectedFace>();
        int rows = faces.Rows;

        for (int i = 0; i < rows; i++)
        {
            int x = (int)faces.At<float>(i, 0);
            int y = (int)faces.At<float>(i, 1);
            int w = (int)faces.At<float>(i, 2);
            int h = (int)faces.At<float>(i, 3);
            float confidence = faces.At<float>(i, 14);

            var landmarks = new List<Point>();
            for (int lm = 0; lm < 5; lm++)
            {
                int lx = (int)faces.At<float>(i, 4 + lm * 2);
                int ly = (int)faces.At<float>(i, 4 + lm * 2 + 1);
                landmarks.Add(new Point(lx, ly));
            }

            result.Add(new DetectedFace
            {
                X = x,
                Y = y,
                Width = w,
                Height = h,
                Confidence = confidence,
                Landmarks = landmarks
            });
        }

        return result;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _detector.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// DTO de rosto detectado
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