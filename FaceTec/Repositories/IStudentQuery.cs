using FaceTec.Util.dataModel;

namespace FaceTec.Repositories;

public abstract class IStudentQueries
{
    public abstract Task<StudentModel?> ObterPorIdAsync(int id, CancellationToken cancellationToken = default);
    public abstract Task<byte[]?> ObterFotoAsync(int studentId, CancellationToken cancellationToken = default);
}