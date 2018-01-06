using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NNTest
{
    class Mnist
    {
       public double[][] images = null;
        public double[]labels = null;
        public Mnist()
        {
            double[][] Images = new double[60000][];
            double[] labels = new double[60000];


            FileStream fs = new FileStream("label.dat", FileMode.Open);
            byte[] buf = new byte[60000];

            fs.Read(buf, 0, 8);
            fs.Read(buf, 0, 60000);
            fs.Close();

            labels = buf.Select(x => (double)x).ToArray();

            fs = new FileStream("image.dat", FileMode.Open);
            byte[] imageBuf = new byte[28 * 28];
            fs.Read(imageBuf, 0, 16);
            for (int i = 0; i < 60000; i++)
            {
                fs.Read(imageBuf, 0, 28 * 28);
                Images[i] = imageBuf.Select(x => ((double)x) / 255.0).ToArray();




            }
            this.images = Images;
            this.labels = labels;
        }
    }

    class Program
    {
        public static void Main()
        {
           Layer l = new Layer(28*28, 28*28);
           Layer l1 = new Layer(100, 28*28);
           Layer l2 = new Layer(30, 100);
           Layer l3 = new Layer(20, 30);
           Layer l4 = new Layer(15, 20);
            Layer l5 = new Layer(10, 15);
            Layers net=new Layers(new Layer[]{l,l1,l2,l3,l4,l5});
            Mnist mnist=new Mnist();

            int ok = 0;
            for (int k = 0; k < 100; k+=1)
            {
                double rate = 1.0 * ok / 10.0;

                 ok = 0;
                for (int i = 0; i < 1000; i+=1)
            {
                    var input=mnist.images[i];
                var ans =new double[10];
                ans[(int) mnist.labels[i]] = 1;

                int Draw(double[] x)
                {
                    double max = -1000;
                    int maxj = 0;
                    for (int j = 0; j < 10; j++)
                    {
                        if (x[j] > max)
                        {
                            max = x[j];
                            maxj = j;
                        }
                    }
                    return maxj;
                }

                int num = Draw(net.Compute(input));
             if(i%100==0)   Console.WriteLine(k+" "+  "input="+mnist.labels[i]+" output="+num +(num== (int)mnist.labels[i]?" ok":" NG")+"正答率="+rate);
                if( (num == (int)mnist.labels[i] )) ok++;
                    net.Lean(input,ans);
            }
                Console.WriteLine(k);
            }

           
        }
    }

    class Layers
    {
        private Layer[] _layers = null;

        public Layers(Layer[] layers)
        {
            this._layers = layers;
        }

        public double[] Compute(double[] inputs)
        {
            var tmp =(double[]) inputs.Clone();
            foreach (var layer in _layers)
            {
                tmp = layer.Compte(tmp);
            }
            return tmp;
        }

        public void Lean(double[] inputs, double[] ansers)
        {
            var tmp = (double[]) inputs.Clone();
            double[][] outputsArray = new double[this._layers.Length][];
            for (int i = 0; i < this._layers.Length - 1; i++)
            {
                tmp = this._layers[i].Compte(tmp);
                outputsArray[i] = (double[]) tmp.Clone();
            }
            outputsArray[this._layers.Length - 1] = (double[]) this._layers.Last().Compte(tmp).Clone();

            //出力層
            this._layers[this._layers.Length - 1].Lean(ansers, outputsArray[this._layers.Length - 1],null);


            for (int i = this._layers.Length - 2; i >= 0; i--)
            {
                    this._layers[i].Lean(_layers[i+1], outputsArray[i],null);

            }

            Parallel.For(0, this._layers.Length, i =>
            {
                if (i != 0)
                    this._layers[i].ApplyDelta(outputsArray[i - 1]);
                else
                    this._layers[i].ApplyDelta(inputs);
            });
        }
    }


    class Layer
    {
        public int PerceptronLength = 0;
        public Perceptron[] Perceptrons;

        public Layer(int PerceptronLength,int inputs)
        {
            this.PerceptronLength = PerceptronLength;
            this.Perceptrons=new Perceptron[this.PerceptronLength];
            for (int i = 0; i < Perceptrons.Length; i++)
            {
                this.Perceptrons[i]=new Perceptron(inputs);
            }
        }

        public double[] Compte(double[] inputs)
        {
            var rtn = new double[PerceptronLength];
            Parallel.For(0, PerceptronLength, i =>
            {
                rtn[i] = this.Perceptrons[i].Compute(inputs);
            });
            return rtn;
        }

        public void Lean(Layer nextLayer, double[] outputs,double[] bfors)
        {
            for (int i = 0; i < this.PerceptronLength; i++)
                this.Perceptrons[i].ComputeDelta(nextLayer, i, outputs[i],0);
        }

        public void Lean(double[] answer, double[] outputs,double[] bfors)
        {
            for (int i = 0; i < this.PerceptronLength; i++)
                this.Perceptrons[i].ComputeDeltaOutput(answer[i], outputs[i], 0);
        }

        public void ApplyDelta(double[] outputs)
        {
            for (int i = 0; i < this.PerceptronLength; i++)
                this.Perceptrons[i].ApplyDelta(outputs);
        }
    }

    static class MyRandom
    {
        public static Random r=new Random();
    }
    

    class Perceptron
    {
        private int InputCount = 0;
        public double[] Weights = null;
        public double Deltas = 0;
        public Perceptron(int inputCount)
        {
            this.InputCount = inputCount;
            this.Weights=new double[this.InputCount+1];
            this.Deltas=0;
            for (int i = 0; i < this.InputCount+1; i++)
            {
                this.Weights[i] =1*( MyRandom.r.NextDouble()-0.5);
            }
        }

        public double sum = 0;
        public double Compute(double[] inputs)
        {
             sum = 0;
            for (int i = 0; i < InputCount; i++)
            {
                sum += this.Weights[i] * inputs[i];
            }
            //bias
           sum += this.Weights[InputCount ];

        var rtn = Math.Tanh(sum);//tanh
        //   var rtn = sum > 0 ? sum*1 : sum*0.01;//reru
            return rtn;
        }

        public void ApplyDelta(double[] output)
        {
            for (int i = 0; i < this.InputCount; i++)
            {
                this.Weights[i]-= 0.01 * this.Deltas * output[i];
            }
            this.Weights[this.InputCount] -= 0.01 * this.Deltas;
            this.Deltas = 0;
        }

        public double scrt(double output)
        {
            var tmp = Math.Exp(output) + Math.Exp(-output);
            return 4.0 / (tmp * tmp);
        }

        public double invTanh(double x)
        {
            return -0.5*Math.Log(1 - x) +0.5* Math.Log(1 + x);
        }

        public void ComputeDeltaOutput(double anser,double output,double bforF)
        {
       this.Deltas = ( output-anser) * scrt(sum);//tanh
      //     this.Deltas = (output - anser) * sum > 0 ? 1 : 0.01; //leru

        }

        public void ComputeDelta(Layer r, int no, double output, double bforF)
        {
            double _sum = 0;
            for (int i = 0; i < r.PerceptronLength; i++)
            {
                _sum += r.Perceptrons[i].Deltas * r.Perceptrons[i].Weights[no];
            }
             var delta = _sum  *scrt(this.sum);//tanh
        //   var delta = _sum * this.sum > 0 ? 1 : 0.01; //reru
            this.Deltas = delta;
        }
    }
}