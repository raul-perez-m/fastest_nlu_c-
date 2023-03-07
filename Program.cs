// See https://aka.ms/new-console-template for more information
using fastest_nlu_c_;
using System.Diagnostics;
using System.Reflection.Metadata.Ecma335;
using System.Text.Json;

Classification execFn(Neural net, List<Test> data)
{
    var good = 0;
    foreach (var test in data)
    {
        var classifications = net.Run(test.utterance);
        if (classifications[0].intent == test.intent)
        {
            good++;
        }
    }
    return new Classification { Good = good, Total = data.Count };
}

void measureCorpus(Corpus[] corpus)
{
    var testData = new List<Test>();
    foreach (var c in corpus)
    {
        foreach (var test in c.tests)
        {
            testData.Add(new Test { intent = c.intent, utterance = test });
        }
    }

    var net = new Neural(new NeuralSettings { log = true});
    var watch = new Stopwatch();
    watch.Start();
    net.Train(corpus);
    watch.Stop();
    Console.WriteLine("Time for training: {0}", watch.Elapsed);
    var result = execFn(net, testData);
    Console.WriteLine("Accuracy: {0}", (result.Good * 100) / result.Total);
    var bench = new Benchmark(duration: null,transactionsPerRun: testData.Count);
    var benchResult = bench.MeasureTransactions(execFn, net, testData);
    Console.WriteLine("Transactions per seconds {0}", benchResult);

}


var jsonData = File.ReadAllText(Path.Combine(Environment.CurrentDirectory, "corpus-massive-en.json"));
var corpusComplete = JsonSerializer.Deserialize<CorpusComplete>(jsonData);
measureCorpus(corpusComplete.data);