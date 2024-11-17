using Microsoft.ML.Data;

namespace IA_ClasificationSpamNoSpam.Representations;

public class Prediccion
{
    [ColumnName("PredictedLabel")]
    public bool IsSpam { get; set; }
    public float Score { get; set; }
}
