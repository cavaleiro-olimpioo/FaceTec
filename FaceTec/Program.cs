using OpenCvSharp;
using System.Diagnostics;
using DotNetEnv;
using FaceTec.Services;
using System.Threading.Tasks;

// 1. Carrega as variáveis de ambiente do arquivo .env
string envPath = Path.GetFullPath(
    Path.Combine(AppContext.BaseDirectory, "../../../../.env")
);
Env.Load(envPath);

string rtsp_path = Environment.GetEnvironmentVariable("url_rtsp")
                   ?? throw new Exception("Variável url_rtsp não encontrada no .env");

// 2. Configurações de resolução do vídeo
int width = 1354;
int height = 780;
int frameSize = width * height * 3; // 3 canais (BGR)

// Caminho para o modelo YuNet atualizado
string modelPath = Path.Combine(AppContext.BaseDirectory, "face_detection_yunet_2023mar.onnx");

// 3. Inicializa o serviço de detecção de rostos (já otimizado internamente)
using var faceService = new GetFaceService(modelPath, width, height);

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

    if (hasNewFrame)
    {
        // Converte os bytes puros para o formato Mat do OpenCV
        using var frame = Mat.FromPixelData(new[] { height, width }, MatType.CV_8UC3, processBuffer);

        // Encontra rostos e desenha no próprio frame (Otimizado com Downscale internamente)
        var processed = faceService.DrawFaces(frame);

        // Exibe a janela
        Cv2.ImShow("Câmera", processed);
    }

    // Pressione 'q' para sair
    if (Cv2.WaitKey(1) == 'q')
    {
        isRunning = false;
        break;
    }
}

// 7. Encerramento seguro
process.Kill();
readTask.Wait(1000); // Aguarda a thread de leitura fechar
Cv2.DestroyAllWindows();