import tensorflow as tf
import numpy as np
class ModelLoader():
    def __init__(self):
        self.lineage_vocab_size = 3982
        self.lineage_length = 19
        self.amp_length=40
        return
    
    def create_classifier(self, path="~/DreamWalker/model_weights"):
        encoder_inputs = tf.keras.layers.Input(shape=(1024,))
        x = tf.keras.layers.RepeatVector(self.lineage_length, name="RepeatVector")(encoder_inputs)
        x = tf.keras.layers.GRU(1024, return_sequences=True, dropout=0.2, name="GRU0")(x)
        x = tf.keras.layers.GRU(1024, return_sequences=True, dropout=0.2, name="GRU1")(x)
        x = tf.keras.layers.Dense(self.lineage_vocab_size, activation="softmax")(x)
        classifier = tf.keras.models.Model(encoder_inputs, x)
        classifier.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                        loss='sparse_categorical_crossentropy',metrics=["accuracy"])
        for i, layer in enumerate(classifier.layers):
            weights = np.load(f"{path}/ClassifierWeights/layer_{i}_weights.npz", allow_pickle=True)["weights"]
            layer.set_weights(weights)
            layer.trainable = False
        return classifier

    def create_generator(self, model_choice="decoder"):
        # Load Classifier
        classifier = self.create_classifier()
        MarkerRepresentModule = tf.keras.models.Model(classifier.layers[0].input,
                                                    classifier.layers[-1].output,
                                                    name="ClassifierModule")
        
        if model_choice=="decoder": # Load Decoder
            latent_dim = 100
            autoencoder = tf.keras.models.load_model('DreamWalker/model_weights/PeptideAutoencoder.keras')
            decoder = tf.keras.models.Model(inputs=autoencoder.layers[6].input,
                                            outputs=autoencoder.layers[-1].output)
        else: # Build Generator
            latent_dim = 32
            decoder = tf.keras.models.load_model('DreamWalker/model_weights/GANWeights/PeptideGenerator.keras')
        inputs = tf.keras.layers.Input(shape=(1024),name="SeqInput")
        x = MarkerRepresentModule(inputs)
        x = tf.keras.layers.Conv1D(128, 4, activation='relu', name="Conv1D")(x)
        x = tf.keras.layers.Flatten(name="Flatten_0")(x)
        x = tf.keras.layers.Dense(latent_dim, activation="tanh")(x)
        x = decoder(x)
        generator = tf.keras.models.Model(inputs, x, name="DreamWalker")
        return generator
    
    def create_oracle(self, path="~/DreamWalker/model_weights/OracleWeights"):
        inputs0 = tf.keras.layers.Input((self.amp_length, 43),name="SeqInput")
        inputs1 = tf.keras.layers.Input((1024,),name="BacteriaInput")
        # Extract Peptide Features
        x0 = tf.keras.layers.Conv1D(128, 5, activation='relu', name="Conv1D_0")(inputs0) # kernel_size=5 works well
        x0 = tf.keras.layers.Conv1D(128, 5, activation='relu', name="Conv1D_1")(x0) # Just two layers work better
        x0 = tf.keras.layers.Flatten(name="Flatten_CNN")(x0)
        x0 = tf.keras.layers.Dense(512, activation="relu", name="CNN_Dense0")(x0)

        # Target Marker Gene Representation
        classifier = self.create_classifier()
        MarkerRepresentModule = tf.keras.models.Model(classifier.layers[0].input,
                                                    classifier.layers[-1].output,
                                                    name="ClassifierModule")
        x1 = MarkerRepresentModule(inputs1)
        x1 = tf.keras.layers.Conv1D(128, 4, activation='relu', name="Conv1D_GRU0")(x1)
        x1 = tf.keras.layers.Flatten(name="Flatten_GRU")(x1)
        x1 = tf.keras.layers.Dense(512, activation="relu", name="GRU_DenseLast")(x1) # mimic the previous version

        # FCN
        x = tf.keras.layers.Concatenate(axis=1, name="Concat_FCN")([x0, x1])
        x = tf.keras.layers.Dense(1024, activation="relu", name="FCN_Dense0")(x)
        x = tf.keras.layers.LayerNormalization(name="LayerNorm_0")(x)
        x = tf.keras.layers.Dense(512, activation="relu", name="FCN_Dense1")(x)
        x = tf.keras.layers.LayerNormalization(name="LayerNorm_1")(x)
        x = tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus, name="Output")(x)
        oracle = tf.keras.models.Model([inputs0, inputs1], x, name="Oracle")
        oracle.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=tf.keras.losses.MeanAbsoluteError(),
                    metrics=[tf.keras.metrics.R2Score()])
        oracle = self.create_oracle()
        for i, layer in enumerate(oracle.layers):
            if i == 3:
                continue
            param = layer.get_weights()
            if len(param) == 0:
                continue
            weights = np.load(f"{path}/layer_{i}_weights.npz")["weights"]
            biases = np.load(f"{path}/layer_{i}_biases.npz")["biases"]
            layer.set_weights([weights, biases])
        return oracle

class Preprocessing():
    def __init__(self, k_size=6):
        self.k_size = k_size
        kmers = self.ref_kmers("", self.k_size)
        self.vectorizer = CountVectorizer(vocabulary = kmers)
        self.seqs = []

    def ref_kmers(self, current_kmer, current_depth):
        if current_depth == 1:
            return [current_kmer+"a",current_kmer+"u",current_kmer+"c",current_kmer+"g"]
        else:
            ret = self.ref_kmers(current_kmer+"a",current_depth-1)
            for nt in ['u','c','g']:
                ret += self.ref_kmers(current_kmer+nt,current_depth-1)
            return ret

    def seq2kmer(self, seq, k):
        kmer = ""
        for i in range(0,len(seq)-k,1):
            kmer += seq[i:i+k]+" "
        return kmer[:-1]

    def CountKmers(self,seqs):
        if type(seqs) in [type([]),type(pd.core.series.Series([1]))]:
            kmer = pd.Series(seqs).apply(lambda x: self.seq2kmer(x, self.k_size))
            transformed_X = self.vectorizer.transform(kmer).toarray()
            return transformed_X
        else:
            raise ValueError("Invalid 'seqs' format. Expected formats are 'list' or 'pandas.core.series.Series'.")

    def ReadFASTA(self,filename,as_pd=True):
        if filename.split(".")[-1] not in ["fasta","fna","fa"]:
            raise ValueError('Invalid file format. Expected formats are ["fasta","fna","fa"].')
        file_handle = open(filename,"r")
        seqs = []
        seqid = []
        tmp_seq = ""
        for line in file_handle:
            if (line[0] == ">"):
                if tmp_seq != "":
                    seqs.append(tmp_seq)
                seqid.append(line.split("\n")[0][1:])
                tmp_seq = ""
            else:
                tmp_seq+=line.split("\n")[0]
        seqs.append(tmp_seq)
        file_handle.close()
        if as_pd:
            fasta = {}
            for i in range(len(seqs)):
                fasta[seqid[i]] = seqs[i]
            return pd.DataFrame(fasta,index=["sequence"]).transpose()["sequence"]
        else:
            return seqs, seqid