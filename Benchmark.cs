using System.Diagnostics;

namespace fastest_nlu_c_
{
    public class Benchmark
    {

        public double duration { get; set; }
        public double transactionsPerRun { get; set; }
        public Benchmark(double? duration, double? transactionsPerRun)
        {
            this.duration = (double)(duration != null ? duration : 10000);
            this.transactionsPerRun = (double)(transactionsPerRun != null ? transactionsPerRun : 1);
        }


        public double MeasureTransactions(Func<Neural, List<Test>, Classification> fn, Neural net, List<Test> data )
        {
            double runs = 0;
            TimeSpan elapsed = new();
            var watch = new Stopwatch();
            watch.Start();
            while (elapsed.TotalMilliseconds < this.duration)
            {
                fn(net, data);
                runs += 1;
                elapsed = watch.Elapsed;
            }
            watch.Stop();
            var timePerTransaction = elapsed.TotalMilliseconds / (runs * this.transactionsPerRun);
            Console.WriteLine("Number of runs {0}", runs);
            return 1000 / timePerTransaction;
        }
    }
}
