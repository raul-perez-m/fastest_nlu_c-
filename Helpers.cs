using System.Globalization;
using System.Text;

namespace fastest_nlu_c_
{
    public static class Helpers
    {
        public static string Normalize(string s)
        {
            var stFormD = s.Normalize(NormalizationForm.FormD);
            var sb = new StringBuilder();

            for (int ich = 0; ich < stFormD.Length; ich++)
            {
                UnicodeCategory uc = CharUnicodeInfo.GetUnicodeCategory(stFormD[ich]);
                if (uc != UnicodeCategory.NonSpacingMark)
                {
                    sb.Append(stFormD[ich]);
                }
            }

            return sb.ToString().ToLower();
        }
        public static string[] Tokenize(string s)
        {
            var split = s.Split(new char[] { ' ', ',', '.', '!', '?', ';', ':', ']', '(', '[', ')', '"', '¡', '¿', '/', '\u0027' }, StringSplitOptions.RemoveEmptyEntries);
            return split;
        }

        public static void DefaultLogFn(Status status ,long time)
        {
            Console.WriteLine("Epoch {0} loss {1} time {2}ms", status.iterations, status.error, time);
        }

        public static double RunInputPerceptron(double[] weights, List<int> input)
        {
            double sum = 0;
            for (int x = 0; x < input.Count; x += 1)
            {
                sum += weights[input[x]];
            }
            return sum <= 0 ? 0 : sum;
        }
    }


    public class EncodeOutput
    {

        public int output { get; set; }
        public List<int> input { get; set; }

    }

    public class Corpus
    {
        public string intent { get; set; }
        public string[] utterances {
            get;
            set;
        }
        public string[] tests {
            get;
            set;
        }
    }

    public class EncodeCorpusResult
    {
        public List<EncodeOutput> train = new();
        public List<EncodeOutput> validation = new();

        public EncodeCorpusResult() { }
        public EncodeCorpusResult(List<EncodeOutput> train, List<EncodeOutput> validation)
        {
            this.train = train;
            this.validation = validation;
        }
    }

    public class NeuralSettings
    {
        public int maxIterations;
        public double learningRate;
        public bool log;
        public Encoder? encoder;
        public Func<string, string[]>? processor;


        public NeuralSettings()
        {
            maxIterations = 150;
            learningRate = 0.002;
            log = false;
        }
        public NeuralSettings(int maxIterations, double learningRate, bool log)
        {
            this.maxIterations = maxIterations;
            this.learningRate = learningRate;
            this.log = log;
        }
    }

    public class Test
    {
        public string utterance;
        public string intent;
    }

    public class Result
    {
        public string intent;
        public double score;
    }

    public class Status
    {
        public double error = 0;
        public int iterations = 0;
    }

    public class Perceptron
    {
        public string intent;
        public int id;
        public double[] weights;
    }

    public class Classification
    {
        public double Good;
        public double Total;
    }

    public class CorpusComplete
    {
        public string name { get; set; }
        public string locale { get; set; }
        public Corpus[] data { get; set; }
    }
}
