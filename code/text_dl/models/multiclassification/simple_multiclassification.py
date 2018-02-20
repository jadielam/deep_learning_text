from text_dl.models import Model
from text_dl.modules.encoders import EncoderRNN
from text_dl.modules.decoders import AttentionDecoder
from text_dl.modules.classification import Classifier


class SimpleMulticlassificationModel(Model):
    def __init__(self, vocab, nb_classes, max_sequence_length):
        super(SimpleMulticlassificationModel, self).__init__()
        self.max_sequence_length
        self.encoder = EncoderRNN(vocab, train_embedding = False, 
                        bidirectional = True)
        
        hidden_size = vocab.vectors.shape[1]
        classifier_input_size = max_sequence_length * hidden_size * 2
        self.classifier = Classifier(nb_classes = nb_classes, clasifier_input_size)
    
    def forward(self, input_t):
        encoder_outputs = Variable(torch.zeros(self.max_sequence_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda() else encoder_outputs
        encoder_hidden = encoder.init_hidden(self.use_cuda())

        input_length = input_t.size()[0]
        for ei in range(min(input_length, self.max_sequence_length)):
            encoder_output, encoder_hidden = encoder(
                input_t
            )

    def loss(self):
        pass
    