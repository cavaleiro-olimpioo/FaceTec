using Dapper;
using FaceTec.Util.dataModel;
using Npgsql;

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
            new NpgsqlConnection(_connectionString);

        

        try
        {
            await conexao.OpenAsync(cancellationToken);

            

            var aluno =
                await conexao.QuerySingleOrDefaultAsync<StudentModel>(
                    new CommandDefinition(
                        commandText: sql,
                        parameters: new { Id = id },
                        cancellationToken: cancellationToken));

            return aluno;
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine(
                "[DB] Operação cancelada.");

            throw;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"[DB] Erro ao buscar aluno {id}:");

            Console.Error.WriteLine(ex);

            throw;
        }
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
            new NpgsqlConnection(_connectionString);

        try
        {
            

            await conexao.OpenAsync(cancellationToken);
            var resultado =
                await conexao.ExecuteScalarAsync<byte[]?>(
                    new CommandDefinition(
                        commandText: sql,
                        parameters: new { Id = id },
                        cancellationToken: cancellationToken));
            return resultado;
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine(
                "[DB] Operação cancelada.");

            throw;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"[DB] Erro ao buscar foto do aluno {id}:");

            Console.Error.WriteLine(ex);

            throw;
        }
    }
}