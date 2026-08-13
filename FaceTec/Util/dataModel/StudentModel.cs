namespace FaceTec.Repositories;

public class StudentModel
{
    public int id { get; set; }
    public string nome { get; set; }
    public string curso { get; set; }
    public string periodo { get; set; }
    public string instituicao { get; set; }
    public byte[] foto_perfil { get; set; }
}