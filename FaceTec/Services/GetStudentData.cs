using FaceTec.Util.dataModel;

namespace FaceTec.Services;

using Dapper;
using Microsoft.Data.SqlClient;

public class GetStudentData
{
    private readonly string _connectionString;

    public GetStudentData(string connectionString)
    {
        _connectionString = connectionString;
    }

    public async Task<StudentModel?> GetStudentDataById(
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
                           WHERE id = @id;
                           """;
        await using var conexao = new SqlConnection(_connectionString);

        return await conexao.QuerySingleOrDefaultAsync<StudentModel>(
            new CommandDefinition(
                sql,
                new { id = id },
                cancellationToken: cancellationToken));
    }

    public async Task<byte[]?> GetStudentPicture(
        int id,
        CancellationToken cancellationToken = default)
    {
        const string sql = """
                           SELECT foto_perfil
                           FROM aluno
                           WHERE id = @id;
                           """;

        await using var conexao = new SqlConnection(_connectionString);

        return await conexao.ExecuteScalarAsync<byte[]>(
            new CommandDefinition(
                sql,
                new { id = id },
                cancellationToken: cancellationToken));
    }
    
}