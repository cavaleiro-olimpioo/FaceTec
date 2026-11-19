using OpenCvSharp;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using DotNetEnv;
using FaceTec.Services;

// 1. Carrega as variáveis de ambiente do arquivo .env
LoadEnvironmentFile();

// 2. Configurações de resolução do vídeo
int width = 1354;
int height = 780;
int frameSize = width * height * 3; // 3 canais (BGR)

int source = ReadVideoSource();
string? rtspPath = source == 2
        ? Environment.GetEnvironmentVariable("url_rtsp")
            ?? throw new Exception("Variável url_rtsp não encontrada no .env")
        : null;

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
bool testMode = ReadTestMode();
string? testPhotosDirectory = testMode ? ResolveTestPhotosDirectory() : null;

// 3. Inicializa o serviço de detecção de rostos (já otimizado internamente)
using var faceService = new GetFaceService(
    modelPath,
    width,
    height,
    connectionString,
    testMode,
    testPhotosDirectory);

// ---- NOVO: assina o evento de reconhecimento UMA ÚNICA VEZ, aqui fora do loop ----
// (assinar dentro do loop de frames criaria um handler novo a cada frame processado)
// Ubuntu: Console.Beep() não existe (lança PlatformNotSupportedException). Usamos o ffplay,
// que já está no PATH porque o RTSP também depende do ffmpeg, pra tocar um tom curto.
// Isso só produz som se a máquina tiver saída de áudio real (PulseAudio/ALSA ativo);
// em servidor headless sem placa de som, nenhum comando resolve — é limite de hardware.
faceService.OnRecognized += _ =>
{
    var beep = new ProcessStartInfo
    {
        FileName = "ffplay",
        Arguments = "-nodisp -autoexit -f lavfi -i \"sine=frequency=1000:duration=0.2\" -loglevel quiet",
        UseShellExecute = false,
        CreateNoWindow = true
    };
    Process.Start(beep);
};

// 4. Inicializa a fonte de vídeo escolhida no terminal.
Process? process = null;
VideoCapture? webcam = null;
Stream? stdout = null;

if (source == 2)
{
    var psi = new ProcessStartInfo
    {
        FileName = "ffmpeg",
        Arguments = $"-rtsp_transport tcp -i \"{rtspPath}\" -vf scale={width}:{height} -f rawvideo -pix_fmt bgr24 -",
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true
    };

    process = Process.Start(psi)
              ?? throw new Exception("Não foi possível iniciar o ffmpeg");
    process.ErrorDataReceived += (_, e) =>
    {
        // if (!string.IsNullOrWhiteSpace(e.Data))
        //     Console.Error.WriteLine($"ffmpeg: {RedactSensitiveUrls(e.Data)}");
    };
    process.BeginErrorReadLine();
    stdout = process.StandardOutput.BaseStream;
}
else
{
    int webcamIndex = ReadWebcamIndex();
    webcam = new VideoCapture(webcamIndex);
    if (!webcam.IsOpened())
        throw new Exception($"Não foi possível abrir a webcam no índice {webcamIndex}. Verifique se ela está conectada e não está sendo usada por outro aplicativo.");

    webcam.Set(VideoCaptureProperties.FrameWidth, width);
    webcam.Set(VideoCaptureProperties.FrameHeight, height);
}

// Variáveis para comunicação entre a thread de leitura e a thread principal
byte[]? sharedBuffer = null;
object bufferLock = new object();
bool isRunning = true;

// 5. THREAD DE LEITURA (Background)
// Lemos o ffmpeg o mais rápido possível para não gerar atrasos (lag) na câmera.
var readTask = Task.Run(() =>
{
    byte[] localBuffer = new byte[frameSize];
    using var webcamFrame = new Mat();
    using var resizedFrame = new Mat();

    while (isRunning)
    {
        if (source == 1)
        {
            if (!webcam!.Read(webcamFrame) || webcamFrame.Empty())
                break;

            Mat frameToCopy = webcamFrame;
            if (webcamFrame.Width != width || webcamFrame.Height != height)
            {
                Cv2.Resize(webcamFrame, resizedFrame, new Size(width, height));
                frameToCopy = resizedFrame;
            }

            Marshal.Copy(frameToCopy.Data, localBuffer, 0, frameSize);
        }
        else
        {
            int byteRead = 0;
            // Lê os bytes até preencher um frame inteiro.
            while (byteRead < frameSize)
            {
                int read = stdout!.Read(localBuffer, byteRead, frameSize - byteRead);
                if (read <= 0) break;
                byteRead += read;
            }

            if (byteRead < frameSize)
                break; // Fim do stream ou erro.
        }

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

        // ---- NOVO: overlay "Bem vindo, Nome" + foto, centralizado no topo ----
        // Entra ANTES do crop pro displayRect: como o crop é simétrico (mesma margem
        // dos dois lados), o texto continua centralizado depois de cortado.
        if (faceService.IsRecognitionActive && faceService.RecognizedName is { } name)
        {
            string firstName = name.Split(' ')[0];
            string text = $"Bem vindo, {firstName}";
            var textSize = Cv2.GetTextSize(text, HersheyFonts.HersheySimplex, 0.9, 2, out _);
            int textX = (processed.Width - textSize.Width) / 2;

            if (faceService.RecognizedPhoto is { } photo)
            {
                using var thumb = new Mat();
                Cv2.Resize(photo, thumb, new Size(80, 80));
                var photoRect = new Rect((processed.Width - 80) / 2, 10, 80, 80);
                thumb.CopyTo(new Mat(processed, photoRect));
            }

            Cv2.PutText(processed, text, new Point(textX, 110),
                HersheyFonts.HersheySimplex, 0.9, Scalar.Lime, 2, LineTypes.AntiAlias);
        }
        // ---- fim do bloco novo ----

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
if (process is not null && !process.HasExited)
    process.Kill();
readTask.Wait(1000); // Aguarda a thread de leitura fechar
webcam?.Release();
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

static void LoadEnvironmentFile()
{
    string legacyEnvPath = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "../../../../.env/.env"));
    if (File.Exists(legacyEnvPath))
    {
        Env.Load(legacyEnvPath);
        return;
    }

    for (DirectoryInfo? directory = new(AppContext.BaseDirectory);
         directory is not null;
         directory = directory.Parent)
    {
        string envPath = Path.Combine(directory.FullName, ".env");
        if (File.Exists(envPath))
        {
            Env.Load(envPath);
            return;
        }
    }
}

static bool ReadTestMode()
{
    string? value = Environment.GetEnvironmentVariable("FACETEC_TEST_MODE");
    if (string.IsNullOrWhiteSpace(value))
        return false;

    if (bool.TryParse(value, out bool enabled))
        return enabled;

    throw new InvalidOperationException("FACETEC_TEST_MODE deve ser true ou false.");
}

static string ResolveTestPhotosDirectory()
{
    string? configuredPath = Environment.GetEnvironmentVariable("FACETEC_TEST_PHOTOS_PATH");
    if (!string.IsNullOrWhiteSpace(configuredPath))
        return Path.GetFullPath(Environment.ExpandEnvironmentVariables(configuredPath));

    for (DirectoryInfo? directory = new(AppContext.BaseDirectory);
         directory is not null;
         directory = directory.Parent)
    {
        string candidate = Path.Combine(directory.FullName, "facesTest");
        if (Directory.Exists(candidate))
            return candidate;
    }

    throw new DirectoryNotFoundException(
        "Pasta de fotos de teste não encontrada. Configure FACETEC_TEST_PHOTOS_PATH.");
}

static int ReadVideoSource()
{
    while (true)
    {
        Console.WriteLine("Escolha a fonte de vídeo:");
        Console.WriteLine("1 - Webcam do notebook");
        Console.WriteLine("2 - RTSP");
        Console.Write("Opção: ");

        string? option = Console.ReadLine()?.Trim();
        if (option is "1" or "2")
            return int.Parse(option);

        Console.WriteLine("Opção inválida. Digite 1 ou 2.\n");
    }
}

static int ReadWebcamIndex()
{
    string? configuredIndex = Environment.GetEnvironmentVariable("FACETEC_CAMERA_INDEX");
    if (string.IsNullOrWhiteSpace(configuredIndex))
        return 0;

    if (int.TryParse(configuredIndex, out int webcamIndex) && webcamIndex >= 0)
        return webcamIndex;

    throw new InvalidOperationException("FACETEC_CAMERA_INDEX deve ser um inteiro maior ou igual a zero.");
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