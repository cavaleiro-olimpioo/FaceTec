namespace FaceTec.Services;

using OpenCvSharp;

/// <summary>
/// Serviço responsável por alinhar rostos no formato canônico esperado por modelos ArcFace.
/// </summary>
/// <remarks>
/// Modelos como o MobileFaceNet w600k_mbf foram treinados com rostos alinhados: olhos,
/// nariz e boca aparecem sempre nas mesmas posições relativas dentro de uma imagem 112x112.
/// Usar apenas o recorte da bounding box preserva rotação, inclinação e escala do frame,
/// produzindo embeddings pouco discriminativos. Este serviço corrige essa variação antes
/// da geração do embedding.
/// </remarks>
public class FaceAlignmentService
{
    /// <summary>
    /// Largura da imagem canônica esperada pelo modelo ArcFace.
    /// </summary>
    public const int AlignedWidth = 112;

    /// <summary>
    /// Altura da imagem canônica esperada pelo modelo ArcFace.
    /// </summary>
    public const int AlignedHeight = 112;

    private static readonly Point2f[] ArcFaceTemplate =
    {
        new(38.2946f, 51.6963f),
        new(73.5318f, 51.5014f),
        new(56.0252f, 71.7366f),
        new(41.5493f, 92.3655f),
        new(70.7299f, 92.2041f)
    };

    /// <summary>
    /// Alinha um rosto usando os 5 landmarks do YuNet na ordem ArcFace.
    /// </summary>
    /// <param name="image">Frame ou imagem original, em resolução cheia.</param>
    /// <param name="landmarks">Landmarks na ordem: olho esquerdo, olho direito, nariz, boca esquerda, boca direita.</param>
    /// <returns>Uma nova imagem 112x112 já alinhada para geração de embedding.</returns>
    /// <exception cref="ArgumentException">Lançada quando a imagem está vazia ou a quantidade de landmarks não é 5.</exception>
    /// <exception cref="InvalidOperationException">Lançada quando o OpenCV não consegue estimar a transformação.</exception>
    public Mat Align(Mat image, IReadOnlyList<Point2f> landmarks)
    {
        if (image.Empty())
            throw new ArgumentException("A imagem usada para alinhamento está vazia.", nameof(image));

        if (landmarks.Count != ArcFaceTemplate.Length)
            throw new ArgumentException("O alinhamento ArcFace exige exatamente 5 landmarks.", nameof(landmarks));

        using var sourcePoints = Mat.FromArray(landmarks.ToArray());
        using var targetPoints = Mat.FromArray(ArcFaceTemplate);
        using var inliers = new Mat();
        var estimatedTransform = Cv2.EstimateAffinePartial2D(
            sourcePoints,
            targetPoints,
            inliers,
            RobustEstimationAlgorithms.LMEDS
        );

        if (estimatedTransform is null || estimatedTransform.Empty())
            throw new InvalidOperationException("Não foi possível estimar a transformação de alinhamento facial.");

        using var transform = estimatedTransform;
        var alignedFace = new Mat();
        Cv2.WarpAffine(
            image,
            alignedFace,
            transform,
            new Size(AlignedWidth, AlignedHeight),
            InterpolationFlags.Linear,
            BorderTypes.Constant,
            Scalar.Black
        );

        return alignedFace;
    }
}
