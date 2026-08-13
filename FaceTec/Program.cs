using OpenCvSharp;
using System.Diagnostics;
using System.Text.RegularExpressions;
using DotNetEnv;
using FaceTec.Services;

// 1. Carrega as variáveis de ambiente do arquivo .env
string envPath = Path.GetFullPath(
    Path.Combine(AppContext.BaseDirectory, "../../../../.env/.env")
);
Env.Load(envPath);

string rtsp_path = Environment.GetEnvironmentVariable("url_rtsp")
                   ?? throw new Exception("Variável url_rtsp não encontrada no .env");

// 2. Configurações de resolução do vídeo
int width = 1354;
int height = 780;
int frameSize = width * height * 3; // 3 canais (BGR)

// Dados da conexão com database
string connectionString =
    "Server=localhost,1433;" +
    "Database=facetec_test;" +
    "User Id=admin;" +
    "Password=admin123;" +
    "TrustServerCertificate=True;";

// Caminho para o modelo YuNet atualizado
string modelPath = Path.Combine(AppContext.BaseDirectory, "face_detection_yunet_2023mar.onnx");
if (!File.Exists(modelPath))
    throw new FileNotFoundException("Modelo YuNet não encontrado.", modelPath);

// FACETEC_SHOW_WINDOW=true/false força o comportamento.
// Sem configuração explícita, abre janela quando houver sessão gráfica disponível.
bool showWindow = ShouldShowWindow();

// 3. Inicializa o serviço de detecção de rostos (já otimizado internamente)
using var faceService = new GetFaceService(modelPath, width, height, connectionString);

// 4. Configura o processo do FFmpeg para capturar o RTSP
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

process.ErrorDataReceived += (_, e) =>
{
    // if (!string.IsNullOrWhiteSpace(e.Data))
        // Console.Error.WriteLine($"ffmpeg: {RedactSensitiveUrls(e.Data)}");
};
process.BeginErrorReadLine();
var stdout = process.StandardOutput.BaseStream;

// Variáveis para comunicação entre a thread de leitura e a thread principal
byte[]? sharedBuffer = null;
object bufferLock = new object();
bool isRunning = true;

// 5. THREAD DE LEITURA (Background)
// Lemos o ffmpeg o mais rápido possível para não gerar atrasos (lag) na câmera.
var readTask = Task.Run(() =>
{
    byte[] localBuffer = new byte[frameSize];
    while (isRunning)
    {
        int byteRead = 0;
        // Lê os bytes até preencher um frame inteiro
        while (byteRead < frameSize)
        {
            int read = stdout.Read(localBuffer, byteRead, frameSize - byteRead);
            if (read <= 0) break;
            byteRead += read;
        }

        if (byteRead < frameSize)
            break; // Fim do stream ou erro

        // Envia o frame recém-lido para a thread principal processar
        lock (bufferLock)
        {
            if (sharedBuffer == null) 
                sharedBuffer = new byte[frameSize];
            
            // Sobrescreve o buffer compartilhado sempre com o frame mais recente
            Buffer.BlockCopy(localBuffer, 0, sharedBuffer, 0, frameSize);
        }
    }
});

// 6. THREAD PRINCIPAL (Processamento e Exibição)
byte[] processBuffer = new byte[frameSize];


// Configuração da janela
float targetRatio = 10f / 16f;
int cropWidth = (int)(height * targetRatio);
int cropX = (width - cropWidth) / 2; // Centraliza horizontalmente

var displayRect = new Rect(cropX, 0, cropWidth, height);

while (isRunning)
{
    bool hasNewFrame = false;

    // Tenta pegar o frame mais recente disponibilizado pela thread de leitura
    lock (bufferLock)
    {
        if (sharedBuffer != null)
        {
            Buffer.BlockCopy(sharedBuffer, 0, processBuffer, 0, frameSize);
            sharedBuffer = null; // Marca como consumido
            hasNewFrame = true;
        }
    }

    if (!hasNewFrame && readTask.IsCompleted)
    {
        isRunning = false;
        break;
    }

    if (hasNewFrame)
    {
        // Converte os bytes puros para o formato Mat do OpenCV
        using var frame = Mat.FromPixelData(new[] { height, width }, MatType.CV_8UC3, processBuffer);

        // Encontra rostos e desenha no próprio frame (Otimizado com Downscale internamente)
        var processed = faceService.DrawFaces(frame);

        using var displayFrame = new Mat(processed, displayRect);
        
        // Exibe a janela
        if (showWindow)
            Cv2.ImShow("Câmera", displayFrame);
    }

    // Pressione 'q' para sair
    if (showWindow && Cv2.WaitKey(1) == 'q')
    {
        isRunning = false;
        break;
    }
}

// 7. Encerramento seguro
if (!process.HasExited)
    process.Kill();
readTask.Wait(1000); // Aguarda a thread de leitura fechar
if (showWindow)
    Cv2.DestroyAllWindows();

static bool ShouldShowWindow()
{
    string? configuredValue = Environment.GetEnvironmentVariable("FACETEC_SHOW_WINDOW");
    if (bool.TryParse(configuredValue, out bool configured))
        return configured;

    return !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("DISPLAY"))
           || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("WAYLAND_DISPLAY"));
}

static string RedactSensitiveUrls(string message)
{
    return Regex.Replace(
        message,
        @"rtsp://[^'""\s]+",
        "rtsp://<redacted>",
        RegexOptions.IgnoreCase
    );
}
