using System.Diagnostics;
using System.Runtime.InteropServices;

namespace fastest_nlu_c_
{
    public class Neural
    {

        public Action<Status, long>? logFn;
        public NeuralSettings settings;
        public Encoder encoder;
        public EncodeCorpusResult encoded;
        public Status status;
        public List<Perceptron> perceptrons = new List<Perceptron>();
        public Helpers helpers = new();

        public Neural(NeuralSettings? settings)
        {
            this.settings = settings != null ? settings : new NeuralSettings();
            logFn = this.settings.log ? helpers.DefaultLogFn : null;
            encoder = this.settings.encoder ?? new Encoder(this.settings.processor);
        }

        public void PrepareCorpus(Corpus[] corpus)
        {
            encoded = encoder.EncodeCorpus(corpus);
        }

        public void Initialize(Corpus[] corpus)
        {
            PrepareCorpus(corpus);
            status = new Status();
            foreach (var intent in encoder.intentMap)
            {
                perceptrons.Add(new Perceptron()
                {
                    id = intent.Value,
                    weights = new double[encoder.numFeature],
                    intent = intent.Key
                });
            }
        }

        public double TrainPerceptron(Perceptron perceptron, List<EncodeOutput> data)
        {
            var weights = perceptron.weights;
            double error = 0;
            for(int i = 0; i < data.Count; i += 1)
            {
                var d = data[i];
                var actualOutput = helpers.RunInputPerceptron(weights, d.input);
                var expectedOutput = d.output == perceptron.id ? 1 : 0;
                var currentError = expectedOutput - actualOutput;
                if(currentError != 0)
                {
                    error += Math.Pow(currentError, 2);
                    var change = currentError * settings.learningRate;
                    foreach (var input in d.input)
                    {
                        weights[input] += change;
                    }
                }
            }
            return error;
        }


        public Status Train(Corpus[] corpus) {
            Initialize(corpus);
            var data = encoded.train;
            while (status.iterations < settings.maxIterations)
            {
                var watch = Stopwatch.StartNew();
                status.iterations++;
                foreach (var perceptron in perceptrons)
                {
                    status.error += TrainPerceptron(perceptron, data);
                }

                status.error = status.error / (data.Count * perceptrons.Count);
                watch.Stop();
                if (logFn != null)
                {
                    logFn(status, watch.ElapsedMilliseconds);
                }
            }
            return status;
        }

        public List<Result> Run(string text)
        {
            var input = encoder.encodeText(text);
            var result = new List<Result>();
            foreach (var perceptron in perceptrons)
            {
                var score = helpers.RunInputPerceptron(perceptron.weights, input);
                if(score != 0)
                {
                    result.Add(new Result { intent= perceptron.intent, score = score });
                }
            }
            if(result.Count == 0)
            {
                result.Add(new Result { intent = "None", score = 0 });
                return result;
            }

            return result.OrderByDescending(x => x.score).ToList();
        }
    }
}
