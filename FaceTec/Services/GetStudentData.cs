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

        Console.WriteLine(
            $"[DB] Buscando dados do aluno {id}...");

        try
        {
            await conexao.OpenAsync(cancellationToken);

            Console.WriteLine("[DB] PostgreSQL conectado!");

            var aluno =
                await conexao.QuerySingleOrDefaultAsync<StudentModel>(
                    new CommandDefinition(
                        commandText: sql,
                        parameters: new { Id = id },
                        cancellationToken: cancellationToken));

            if (aluno == null)
            {
                Console.WriteLine(
                    $"[DB] Aluno {id} não encontrado.");
            }
            else
            {
                Console.WriteLine(
                    $"[DB] Aluno {id}: {aluno.nome}");
            }

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

        Console.WriteLine(
            $"[DB] Buscando foto do aluno {id}...");

        await using var conexao =
            new NpgsqlConnection(_connectionString);

        try
        {
            Console.WriteLine("[DB] Abrindo conexão PostgreSQL...");

            await conexao.OpenAsync(cancellationToken);

            Console.WriteLine(
                "[DB] CONEXÃO POSTGRESQL ABERTA!");

            Console.WriteLine(
                "[DB] Executando SELECT da foto...");

            var resultado =
                await conexao.ExecuteScalarAsync<byte[]?>(
                    new CommandDefinition(
                        commandText: sql,
                        parameters: new { Id = id },
                        cancellationToken: cancellationToken));

            Console.WriteLine(
                resultado == null
                    ? "[DB] Foto NULL"
                    : $"[DB] Foto recebida: {resultado.Length} bytes");

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