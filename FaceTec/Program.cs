using OpenCvSharp;
using System.Diagnostics;
using DotNetEnv;
using FaceTec.Services;

string envPath = Path.GetFullPath(
    Path.Combine(AppContext.BaseDirectory, "../../../../.env")
);

Env.Load(envPath);

string rtsp_path = Environment.GetEnvironmentVariable("url_rtsp")
                   ?? throw new Exception("Variável url_rtsp não encontrada no .env");

int width = 1354;
int height = 780;
int frameSize = width * height * 3;

// Caminho para o modelo YuNet (face_detection_yunet_2022mar.onnx)
string modelPath = "face_detection_yunet_2022mar.onnx";

// Instancia o serviço de detecção de rostos
using var faceService = new GetFaceService(modelPath, width, height);

var psi = new ProcessStartInfo
{
    FileName = "ffmpeg",
    Arguments = $"-rtsp_transport tcp -i \"{rtsp_path}\" -vf scale={width}:{height} -f rawvideo -pix_fmt bgr24 -",
    RedirectStandardOutput = true,
    RedirectStandardError = true,
    UseShellExecute = false,
    CreateNoWindow = true
};

using var process = Process.Start(psi)
                    ?? throw new Exception("Não foi possível iniciar o ffmpeg");

process.BeginErrorReadLine();

var stdout = process.StandardOutput.BaseStream;
byte[] buffer = new byte[frameSize];

while (true)
{
    int byteRead = 0;
    while (byteRead < frameSize)
    {
        int read = stdout.Read(buffer, byteRead, frameSize - byteRead);
        if (read <= 0) break;
        byteRead += read;
    }

    if (byteRead < frameSize)
        break;

    using var frame = Mat.FromPixelData(new[] { height, width }, MatType.CV_8UC3, buffer);

    // Processa o frame e desenha os rostos
    var processed = faceService.DrawFaces(frame);

    Cv2.ImShow("Câmera", processed);

    // Pressione 'q' para sair
    if (Cv2.WaitKey(1) == 'q')
        break;
}

process.Kill();
Cv2.DestroyAllWindows();