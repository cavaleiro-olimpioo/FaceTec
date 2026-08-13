using Dapper;
using FaceTec.Util.dataModel;
using Microsoft.Data.SqlClient;

namespace FaceTec.Services;

public sealed class GetStudentData
{
    private readonly string _connectionString;

    public GetStudentData(string connectionString)
    {
        _connectionString = connectionString;
    }

    public async Task<StudentModel?> GetStudentDataByIdAsync(
        int id,
        CancellationToken cancellationToken = default)
    {
        const string sql = """
                           SELECT
                               id,
                               nome,
                               curso,
                               periodo,
                               instituicao
                           FROM aluno
                           WHERE id = @Id;
                           """;

        await using var conexao =
            new SqlConnection(_connectionString);

        return await conexao.QuerySingleOrDefaultAsync<StudentModel>(
            new CommandDefinition(
                commandText: sql,
                parameters: new { Id = id },
                cancellationToken: cancellationToken));
    }

    public async Task<byte[]?> GetStudentPictureAsync(
        int id,
        CancellationToken cancellationToken = default)
    {
        const string sql = """
                           SELECT foto_perfil
                           FROM aluno
                           WHERE id = @Id;
                           """;

        await using var conexao =
            new SqlConnection(_connectionString);

        return await conexao.ExecuteScalarAsync<byte[]?>(
            new CommandDefinition(
                commandText: sql,
                parameters: new { Id = id },
                cancellationToken: cancellationToken));
    }
}