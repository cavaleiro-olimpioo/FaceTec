using FaceTec.Util.dataModel;

namespace FaceTec.Repositories;

public interface IStudentQueries
{
    Task<StudentModel?> ObterPorIdAsync(
        int id,
        CancellationToken cancellationToken = default);

    Task<byte[]?> ObterFotoAsync(
        int studentId,
        CancellationToken cancellationToken = default);
}