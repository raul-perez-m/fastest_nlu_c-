using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace fastest_nlu_c_
{
    public class Encoder
    {

        public Dictionary<string, int> featureMap;
        public Dictionary<string, int> intentMap;
        public int numFeature = 0;
        private List<string> intents;
        private Func<string, string[]> processor;
        private Helpers helpers = new();

        public Encoder(Func<string, string[]>? processor) {
            this.processor = processor != null ? processor : (str) => helpers.Tokenize(helpers.Normalize(str));
            featureMap = new Dictionary<string, int>();
            intentMap = new Dictionary<string, int>();
            intents = new List<string>();
        }


        public void learnIntent(string intent)
        {
            if (!intentMap.ContainsKey(intent))
            {
                intentMap.Add(intent, intents.Count);
                intents.Add(intent);
            }
        }

        public void learnFeature(string feature)
        {
            if (!featureMap.ContainsKey(feature))
            {
                featureMap.Add(feature, numFeature);
                numFeature++;
            }
        }

        public List<int> encodeText(string text, bool learn = false) {
            var dict = new Dictionary<int, int>();
            var keys = new List<int>();
            var features = processor(text);
            foreach ( var feature in features)
            {
                if(learn)
                {
                    learnFeature(feature);
                }
                int index;
                featureMap.TryGetValue(feature, out index);
                if (featureMap.ContainsKey(feature) && !dict.ContainsKey(index) )
                {
                    dict.Add(index, 1);
                    keys.Add(index);
                }
            }
            return keys;
        }

        public EncodeOutput encode(string text, string intent, bool learn = false)
        {
            if (learn)
            {
                learnIntent(intent);
            }
            return new EncodeOutput { output = intentMap[intent], input = encodeText(text, learn) };
        }

        public EncodeCorpusResult EncodeCorpus(Corpus[] corpus)
        {
            var result = new EncodeCorpusResult();
            foreach ( var c in corpus)
            {
                if (c.utterances.Length > 0)
                {
                    foreach( var utterance in c.utterances)
                    {
                        result.train.Add(encode(utterance, c.intent, true));
                    };
                }
             
            }
            foreach ( var c in corpus)
            {
                if (c.tests.Length > 0)
                {
                    foreach (var test in c.tests)
                    {
                        result.validation.Add(encode(test, c.intent));
                    }
                }
            }
            return result;
        }
    }
}
