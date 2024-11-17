using IA_ClasificationSpamNoSpam.Representations;
using Microsoft.ML;

namespace IA_ClasificationSpamNoSpam.Process;

public class Procesamiento
{
    public void Procesar()
    {
        // Crear un contexto de ML.NET para gestionar el proceso de ML.
        MLContext mlContext = new MLContext();

        // Conjunto de datos más extenso para entrenamiento.
        var trainingData = new List<TextoDato>
        {
            new TextoDato { Texto = "¡Ganaste una Tarjeta de $1000!", IsSpam = true },
            new TextoDato { Texto = "Oferta limitada: Compra uno y llevate uno gratis", IsSpam = true },
            new TextoDato { Texto = "¡Esto no es un simulacro! Reclama tu premio", IsSpam = true },
            new TextoDato { Texto = "Reunión del equipo reprogramada para las 3:00 p. m.", IsSpam = false },
            new TextoDato { Texto = "Factura de su compra reciente adjunta", IsSpam = false },
            new TextoDato { Texto = "Recordatorio: mañana tengo cita con el médico.", IsSpam = false },
            new TextoDato { Texto = "¡Felicitaciones! Has sido seleccionado para una prueba gratuita", IsSpam = true },
            new TextoDato { Texto = "Tu estado de cuenta bancaria está listo", IsSpam = false }
        };

        // Cargar los datos de entrenamiento en formato IDataView.
        IDataView dataView = mlContext.Data.LoadFromEnumerable(trainingData);

        // Pipeline de procesamiento y entrenamiento:
        // - FeaturizeText: Convierte el texto en un vector de características.
        // - SdcaLogisticRegression: Modelo de clasificación para detectar Spam.
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(TextoDato.Texto))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: nameof(TextoDato.IsSpam),
                featureColumnName: "Features"));

        // Entrenar el modelo con los datos.
        var model = pipeline.Fit(dataView);

        // Crear un conjunto de datos de prueba para evaluación y predicción.
        var testData = new List<TextoDato>
        {
            new TextoDato { Texto = "¡Ganaste unas vacaciones gratis para Hawaii!" },
            new TextoDato { Texto = "Revise el documento adjunto y brinde sus comentarios." },
            new TextoDato { Texto = "Urgente: Su cuenta ha sido vulnerada. ¡Actúe ahora!" },
            new TextoDato { Texto = "Las actualizaciones semanales del proyecto están disponibles en la unidad compartida." },
            new TextoDato { Texto = "¡Compra nuestro producto ahora y obtén un 50% de descuento!" }
        };

        // Convertir el conjunto de prueba en IDataView.
        IDataView testDataView = mlContext.Data.LoadFromEnumerable(testData);

        // Crear un predictor para realizar predicciones.
        var predictor = mlContext.Model.CreatePredictionEngine<TextoDato, Prediccion>(model);

        Console.WriteLine("Resultados de la Clasificación:");
        Console.WriteLine("--------------------------------");

        // Generar predicciones para cada texto en el conjunto de prueba.
        foreach (var text in testData)
        {
            var prediction = predictor.Predict(text);
            Console.WriteLine($"Texto: {text.Texto}");
            Console.WriteLine($"¿Es Spam?: {(prediction.IsSpam ? "Sí" : "No")}");
            Console.WriteLine($"Confianza: {prediction.Score:F2}");
            Console.WriteLine();
        }

        // Evaluar el modelo con métricas usando un conjunto de prueba etiquetado.
        var evaluationData = new List<TextoDato>
        {
            new TextoDato { Texto = "¡Ganaste unas vacaciones gratis para Hawaii!", IsSpam = true },
            new TextoDato { Texto = "Revise el documento adjunto y brinde sus comentarios.", IsSpam = false },
            new TextoDato { Texto = "Urgente: Su cuenta ha sido vulnerada. ¡Actúe ahora!", IsSpam = true },
            new TextoDato { Texto = "Las actualizaciones semanales del proyecto están disponibles en la unidad compartida.", IsSpam = false },
            new TextoDato { Texto = "¡Compra nuestro producto ahora y obtén un 50% de descuento!", IsSpam = true }
        };

        var evaluationDataView = mlContext.Data.LoadFromEnumerable(evaluationData);
        var metrics = mlContext.BinaryClassification.Evaluate(model.Transform(evaluationDataView),
            labelColumnName: nameof(TextoDato.IsSpam));

        Console.WriteLine("Evaluación del Modelo:");
        Console.WriteLine($"Precisión: {metrics.Accuracy:P2}");
        Console.WriteLine($"Recall: {metrics.PositiveRecall:P2}");
        Console.WriteLine($"Especificidad: {metrics.NegativeRecall:P2}");
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
        Console.WriteLine($"Log-Loss: {metrics.LogLoss:F4}");
    }    
}
