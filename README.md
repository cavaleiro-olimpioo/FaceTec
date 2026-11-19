# FaceTec

Sistema de reconhecimento facial conectado a um banco de dados, desenvolvido para controle de acesso de alunos em uma escola.

## Descrição

O projeto captura o rosto do aluno via câmera (stream RTSP), realiza a detecção facial com **YuNet**, alinha a face detectada e gera um embedding utilizando **ArcFace**. Esse embedding é comparado por similaridade de cosseno com os rostos previamente cadastrados no banco de dados, permitindo identificar o aluno em tempo real.

A linguagem principal do projeto é **C# (.NET)**.

## Funcionalidades

### Implementadas
- Captura de vídeo via webcam do notebook (dispositivo padrão ou índice configurável)
- Captura de vídeo via RTSP (câmeras estilo Dahua/Intelbras)
- Detecção facial (YuNet) + alinhamento (ArcFace 112x112)
- Geração de embedding facial via modelo ONNX
- Comparação por similaridade de cosseno com o banco de dados

### Planejadas
- **Interface do porteiro**: a cada verificação bem-sucedida, os dados do aluno reconhecido aparecem em uma interface no computador da portaria.
- **Bloqueio por horário**: alunos são liberados apenas no turno em que estudam (ex.: aluno da manhã tentando entrar à tarde é bloqueado). Tentativas bloqueadas geram uma notificação (ex.: grupo do WhatsApp) informando a tentativa.
- **Vigilância dupla**: uma segunda câmera, posicionada mais à frente da entrada, atua como conferência. Os reconhecimentos da primeira câmera ficam em cache temporário; se um aluno for identificado pela segunda câmera sem ter passado pela primeira, um responsável é notificado com os dados do aluno.

> As funcionalidades acima ainda estão em fase de planejamento e não estão implementadas no código atual.

## Como usar

1. Clone o repositório:
   ```bash
   git clone https://github.com/cavaleiro-olimpioo/FaceTec.git
   ```

2. Para usar a webcam do notebook, selecione a opção `1` ao iniciar. Por padrão, o programa abre o dispositivo de índice `0`. Se houver mais de uma câmera, configure outro índice antes de executar:
   ```bash
   FACETEC_CAMERA_INDEX=1 dotnet run --project FaceTec
   ```
   No PowerShell, use `$env:FACETEC_CAMERA_INDEX="1"` antes de executar `dotnet run --project FaceTec`.

3. Para testar com fotos locais em vez das fotos armazenadas no banco, configure no arquivo `.env`:
   ```env
   FACETEC_TEST_MODE=true
   ```
   Coloque as imagens em `FaceTec/facesTest`, nomeadas pelo ID do aluno em sequência, por exemplo `1.jpg`, `2.png` ou `face1.jpeg`, `face2.jpeg`. São aceitos arquivos JPG, JPEG, PNG, BMP e WEBP. A pasta pode ser alterada com `FACETEC_TEST_PHOTOS_PATH`; sem essa variável, o programa procura `facesTest` junto ao projeto. O modo padrão é `false`, que mantém a leitura das fotos no banco.

   O nome e os dados do aluno reconhecido continuam sendo consultados no banco; esse modo substitui somente a origem das fotos.

4. Para usar uma câmera RTSP, selecione a opção `2` e configure a URL no arquivo `.env`:
   ```
   url_rtsp="rtsp://usuario:senha@192.168.0.1:554/cam/realmonitor?channel=1&subtype=0"
   ```

5. Configure a conexão com o banco de dados em `Program.cs`:
   ```csharp
   string connectionString =
       "Server=SEU_SERVIDOR,1433;" +
       "Database=SEU_BANCO;" +
       "User Id=SEU_USUARIO;" +
       "Password=SUA_SENHA;" +
       "TrustServerCertificate=True;";
   ```


6. Execute o projeto normalmente pela IDE (Rider/Visual Studio) ou via `dotnet run`.

## Stack

- **Linguagem:** C# (.NET)
- **Visão computacional:** OpenCvSharp4
- **Detecção facial:** YuNet
- **Reconhecimento facial:** ArcFace (embedding via ONNX, modelo w600k_mbf)
- **Banco de dados:** SQL Server
