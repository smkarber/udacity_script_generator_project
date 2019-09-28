# TV Script Generation
In this project, you'll generate your own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  You'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network you'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
## Get the Data
The data is already provided for you.  You'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
```

## Explore the Data
Play around with `view_sentence_range` to view different parts of the data.


```python
view_sentence_range = (13, 40)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 11492
    Number of scenes: 262
    Average number of sentences in each scene: 15.251908396946565
    Number of lines: 4258
    Average number of words in each line: 11.50164396430249
    
    The sentences 13 to 40:
    
    
    Barney_Gumble: Hey Homer, how's your neighbor's store doing?
    Homer_Simpson: Lousy. He just sits there all day. He'd have a great job if he didn't own the place. (CHUCKLES)
    Moe_Szyslak: (STRUGGLING WITH CORKSCREW) Crummy right-handed corkscrews! What does he sell?
    Homer_Simpson: Uh, well actually, Moe...
    HOMER_(CONT'D: I dunno.
    
    
    Moe_Szyslak: Looks like this is the end.
    Barney_Gumble: That's all right. I couldn't have led a richer life.
    Barney_Gumble: So the next time somebody tells you county folk are good, honest people, you can spit in their faces for me!
    Lisa_Simpson: I will, Mr. Gumbel. But if you'll excuse me, I'm profiling my dad for the school paper. I thought it would be neat to follow him around for a day to see what makes him tick.
    Barney_Gumble: Oh, that's sweet. I used to follow my dad to a lot of bars too. (BELCH)
    Moe_Szyslak: Here you go. One beer, one chocolate milk.
    Lisa_Simpson: Uh, excuse me, I have the chocolate milk.
    Moe_Szyslak: Oh.
    Moe_Szyslak: What's the matter, Homer? The depressin' effects of alcohol usually don't kick in 'til closing time.
    Lisa_Simpson: He's just a little nervous. (PROUDLY) He has to give a speech tomorrow on "How To Keep Cool In A Crisis."
    Homer_Simpson: (SOBS) What am I gonna do? What am I gonna do?
    Barney_Gumble: Hey, I had to give a speech once. I was pretty nervous, so I used a little trick. I pictured everyone in their underwear. The judge, the jury, my lawyer, everybody.
    Homer_Simpson: Did it work?
    Barney_Gumble: I'm a free man, ain't I?
    Barney_Gumble: Whoa!
    Barney_Gumble: Huh? A pretzel? Wow, looks like I pulled a Homer!
    
    
    

## Implement Preprocessing Functions
The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`


```python
import numpy as np
import problem_unittests as tests

from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counter = Counter(text)
    sorted_words = sorted(word_counter, key=word_counter.get, reverse=True)        
    vocab_to_int = {word: ii for ii, word in enumerate(sorted_words)}
    
    int_to_vocab = {ii: word for word, ii in vocab_to_int.items()}
    
    return (vocab_to_int, int_to_vocab)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)
```

    Tests Passed
    

### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( -- )
- Return ( \n )

This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure you don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    return {
      '.': '>>PERIOD<<',
      ',': '>>COMMA<<',
      '"': '>>DOUBLE_QUOTE<<',
      ';': '>>SEMICOLON<<',
      '!': '>>EXCLAMATION_POINT<<',
      '?': '>>QUESTION_MARK<<',
      '(': '>>LEFT_PARENTHESIS<<',
      ')': '>>RIGHT_PARENTHESIS<<',
      '--': '>>DASH<<',
      '\n': '>>RETURN<<'
    }

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)
```

    Tests Passed
    

## Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network
You'll build the components necessary to build a RNN by implementing the following functions below:
- get_inputs
- get_init_cell
- get_embed
- build_rnn
- build_nn
- get_batches

### Check the Version of TensorFlow and Access to GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.3'), 'Please use TensorFlow version 1.3 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.14.0
    Default GPU Device: /device:GPU:0
    

### Input
Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
- Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
- Targets placeholder
- Learning Rate placeholder

Return the placeholders in the following tuple `(Input, Targets, LearningRate)`


```python
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs = tf.placeholder(tf.int32, [None, None], name ='input') 
    targets = tf.placeholder(tf.int32, [None, None], name ='targets')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    
    return inputs, targets, learning_rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)
```

    Tests Passed
    

### Build RNN Cell and Initialize
Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
- The Rnn size should be set using `rnn_size`
- Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
    - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the cell and initial state in the following tuple `(Cell, InitialState)`


```python
def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    num_layers = 2
    keep_prob = 0.5
  
    def build_cell():
        cl = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        return tf.contrib.rnn.DropoutWrapper(cl, output_keep_prob=keep_prob)
        
    cell = tf.contrib.rnn.MultiRNNCell([build_cell()] * num_layers)
    
    initial_state = cell.zero_state(batch_size, tf.float32)    
    
    return cell, tf.identity(initial_state, "initial_state")


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)
```

    WARNING:tensorflow:From D:\Code\Deep-Learning\deep-learning\tv-script-generation\problem_unittests.py:185: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From <ipython-input-10-20e22e871cb7>:12: BasicLSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.
    WARNING:tensorflow:From <ipython-input-10-20e22e871cb7>:15: MultiRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.StackedRNNCells, and will be replaced by that in Tensorflow 2.0.
    WARNING:tensorflow:At least two cells provided to MultiRNNCell are the same object and will share weights.
    Tests Passed
    

### Word Embedding
Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.


```python
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1, dtype = tf.float32))
    embed = tf.nn.embedding_lookup(embedding, input_data)

    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)
```

    Tests Passed
    

### Build RNN
You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
- Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
 - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 


```python
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
    
    return outputs, tf.identity(final_state, "final_state")

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)
```

    WARNING:tensorflow:From <ipython-input-12-e5ca2edaa30f>:8: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.RNN(cell)`, which is equivalent to this API
    WARNING:tensorflow:From c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\ops\init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    WARNING:tensorflow:From c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\ops\rnn_cell_impl.py:738: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    Tests Passed
    

### Build the Neural Network
Apply the functions you implemented above to:
- Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
- Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
- Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.

Return the logits and final state in the following tuple (Logits, FinalState) 


```python
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """    
    inputs = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, inputs)

    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, None)

    return logits, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)
```

    Tensor("fully_connected/BiasAdd:0", shape=(128, 5, 27), dtype=float32)
    Tests Passed
    

### Batches
Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
- The first element is a single batch of **input** with the shape `[batch size, sequence length]`
- The second element is a single batch of **targets** with the shape `[batch size, sequence length]`

If you can't fill the last batch with enough data, drop the last batch.

For example, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)` would return a Numpy array of the following:
```
[
  # First Batch
  [
    # Batch of Input
    [[ 1  2], [ 7  8], [13 14]]
    # Batch of targets
    [[ 2  3], [ 8  9], [14 15]]
  ]

  # Second Batch
  [
    # Batch of Input
    [[ 3  4], [ 9 10], [15 16]]
    # Batch of targets
    [[ 4  5], [10 11], [16 17]]
  ]

  # Third Batch
  [
    # Batch of Input
    [[ 5  6], [11 12], [17 18]]
    # Batch of targets
    [[ 6  7], [12 13], [18  1]]
  ]
]
```

Notice that the last target value in the last batch is the first input value of the first batch. In this case, `1`. This is a common technique used when creating sequence batches, although it is rather unintuitive.


```python
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """    
    words_per_batch = batch_size * seq_length
    num_batches = len(int_text) // words_per_batch   
   
    int_text = int_text[:num_batches * words_per_batch]
    
    batches = np.zeros(shape=(num_batches, 2, batch_size, seq_length), dtype = np.int32)

    for i in range(num_batches):
        iteration = 0
        for j in range(i * seq_length, len(int_text), num_batches * seq_length):            
            chunk_start = j
                                
            chunk = int_text[chunk_start:chunk_start + seq_length]
            
            chunk_target = None
            
            if j + seq_length < len(int_text):
                chunk_target = int_text[chunk_start + 1:chunk_start + seq_length + 1]
            else:
                chunk_target = int_text[chunk_start + 1:chunk_start + seq_length + 1]
                chunk_target.append(int_text[0])
        
            batches[i][0][iteration] = chunk
            batches[i][1][iteration] = chunk_target        
            
            iteration += 1
    
    return batches

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)
```

    Tests Passed
    

## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `num_epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `embed_dim` to the size of the embedding.
- Set `seq_length` to the length of sequence.
- Set `learning_rate` to the learning rate.
- Set `show_every_n_batches` to the number of batches the neural network should print progress.


```python
# Number of Epochs
num_epochs = 15
# Batch Size
batch_size = 100
# RNN Size
rnn_size = 512
# Embedding Dimension Size
embed_dim = 500
# Sequence Length
seq_length = 15
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 5000

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
```

    <tensorflow.python.ops.rnn_cell_impl.MultiRNNCell object at 0x00000203BAFDBF28>
    512
    Tensor("input:0", shape=(?, ?), dtype=int32)
    6780
    500
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-29-82c8fc290bfe> in <module>
         15     print(vocab_size)
         16     print(embed_dim)
    ---> 17     logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)
         18 
         19     # Probabilities for generating words
    

    <ipython-input-25-b2a36e78d0e7> in build_nn(cell, rnn_size, input_data, vocab_size, embed_dim)
         10     """    
         11     inputs = get_embed(input_data, vocab_size, embed_dim)
    ---> 12     outputs, final_state = build_rnn(cell, inputs)
         13 
         14     logits = tf.contrib.layers.fully_connected(outputs, vocab_size, None)
    

    <ipython-input-12-e5ca2edaa30f> in build_rnn(cell, inputs)
          6     :return: Tuple (Outputs, Final State)
          7     """
    ----> 8     outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
          9 
         10     return outputs, tf.identity(final_state, "final_state")
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\util\deprecation.py in new_func(*args, **kwargs)
        322               'in a future version' if date is None else ('after %s' % date),
        323               instructions)
    --> 324       return func(*args, **kwargs)
        325     return tf_decorator.make_decorator(
        326         func, new_func, 'deprecated',
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\ops\rnn.py in dynamic_rnn(cell, inputs, sequence_length, initial_state, dtype, parallel_iterations, swap_memory, time_major, scope)
        705         swap_memory=swap_memory,
        706         sequence_length=sequence_length,
    --> 707         dtype=dtype)
        708 
        709     # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\ops\rnn.py in _dynamic_rnn_loop(cell, inputs, initial_state, parallel_iterations, swap_memory, sequence_length, dtype)
        914       parallel_iterations=parallel_iterations,
        915       maximum_iterations=time_steps,
    --> 916       swap_memory=swap_memory)
        917 
        918   # Unpack final output if not using output tuples.
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\ops\control_flow_ops.py in while_loop(cond, body, loop_vars, shape_invariants, parallel_iterations, back_prop, swap_memory, name, maximum_iterations, return_same_structure)
       3499       ops.add_to_collection(ops.GraphKeys.WHILE_CONTEXT, loop_context)
       3500     result = loop_context.BuildLoop(cond, body, loop_vars, shape_invariants,
    -> 3501                                     return_same_structure)
       3502     if maximum_iterations is not None:
       3503       return result[1]
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\ops\control_flow_ops.py in BuildLoop(self, pred, body, loop_vars, shape_invariants, return_same_structure)
       3010       with ops.get_default_graph()._mutation_lock():  # pylint: disable=protected-access
       3011         original_body_result, exit_vars = self._BuildLoop(
    -> 3012             pred, body, original_loop_vars, loop_vars, shape_invariants)
       3013     finally:
       3014       self.Exit()
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\ops\control_flow_ops.py in _BuildLoop(self, pred, body, original_loop_vars, loop_vars, shape_invariants)
       2935         expand_composites=True)
       2936     pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
    -> 2937     body_result = body(*packed_vars_for_body)
       2938     post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
       2939     if not nest.is_sequence_or_composite(body_result):
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\ops\control_flow_ops.py in <lambda>(i, lv)
       3454         cond = lambda i, lv: (  # pylint: disable=g-long-lambda
       3455             math_ops.logical_and(i < maximum_iterations, orig_cond(*lv)))
    -> 3456         body = lambda i, lv: (i + 1, orig_body(*lv))
       3457 
       3458     if executing_eagerly:
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\ops\rnn.py in _time_step(time, output_ta_t, state)
        882           skip_conditionals=True)
        883     else:
    --> 884       (output, new_state) = call_cell()
        885 
        886     # Keras cells always wrap state as list, even if it's a single tensor.
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\ops\rnn.py in <lambda>()
        868     if is_keras_rnn_cell and not nest.is_sequence(state):
        869       state = [state]
    --> 870     call_cell = lambda: cell(input_t, state)
        871 
        872     if sequence_length is not None:
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\ops\rnn_cell_impl.py in __call__(self, inputs, state, scope)
        246         setattr(self, scope_attrname, scope)
        247       with scope:
    --> 248         return super(RNNCell, self).__call__(inputs, state)
        249 
        250   def _rnn_get_variable(self, getter, *args, **kwargs):
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\layers\base.py in __call__(self, inputs, *args, **kwargs)
        535 
        536       # Actually call layer
    --> 537       outputs = super(Layer, self).__call__(inputs, *args, **kwargs)
        538 
        539     if not context.executing_eagerly():
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\keras\engine\base_layer.py in __call__(self, inputs, *args, **kwargs)
        632                     outputs = base_layer_utils.mark_as_return(outputs, acd)
        633                 else:
    --> 634                   outputs = call_fn(inputs, *args, **kwargs)
        635 
        636             except TypeError as e:
    

    c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\autograph\impl\api.py in wrapper(*args, **kwargs)
        147       except Exception as e:  # pylint:disable=broad-except
        148         if hasattr(e, 'ag_error_metadata'):
    --> 149           raise e.ag_error_metadata.to_exception(type(e))
        150         else:
        151           raise
    

    ValueError: in converted code:
        relative to c:\programdata\anaconda3\envs\tensorflow-gpu\lib\site-packages\tensorflow\python:
    
        ops\rnn_cell_impl.py:1719 call *
            cur_inp, new_state = cell(cur_inp, cur_state)
        ops\rnn_cell_impl.py:1159 __call__
            inputs, state, cell_call_fn=self.cell.__call__, scope=scope)
        ops\rnn_cell_impl.py:766 call *
            gate_inputs = math_ops.matmul(
        util\dispatch.py:180 wrapper
            return target(*args, **kwargs)
        ops\math_ops.py:2647 matmul
            a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
        ops\gen_math_ops.py:6295 mat_mul
            name=name)
        framework\op_def_library.py:788 _apply_op_helper
            op_def=op_def)
        util\deprecation.py:507 new_func
            return func(*args, **kwargs)
        framework\ops.py:3616 create_op
            op_def=op_def)
        framework\ops.py:2027 __init__
            control_input_ops)
        framework\ops.py:1867 _create_c_op
            raise ValueError(str(e))
    
        ValueError: Dimensions must be equal, but are 1024 and 1012 for 'rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1' (op: 'MatMul') with input shapes: [?,1024], [1012,2048].
    


## Train
Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forums](https://discussions.udacity.com/) to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
```

## Save Parameters
Save `seq_length` and `save_dir` for generating a new TV script.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()
```

## Implement Generate Functions
### Get Tensors
Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
- "input:0"
- "initial_state:0"
- "final_state:0"
- "probs:0"

Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 


```python
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    input_tensor = loaded_graph.get_tensor_by_name("input:0")
    initial_state_tensor = loaded_graph.get_tensor_by_name("initial_state:0")
    final_state_tensor = loaded_graph.get_tensor_by_name("final_state:0")
    probabilities_tensor = loaded_graph.get_tensor_by_name("probs:0")
    
    return (input_tensor, initial_state_tensor, final_state_tensor, probabilities_tensor)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)
```

    WARNING:tensorflow:From D:\Code\Deep-Learning\deep-learning\tv-script-generation\problem_unittests.py:275: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    Tests Passed
    

### Choose Word
Implement the `pick_word()` function to select the next word using `probabilities`.


```python
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # choosing a random number from the top 3 best choices
    prob = np.squeeze(probabilities)
    prob[np.argsort(prob)[:-3]] = 0
    prob = prob / np.sum(prob)
    predicted_word = np.random.choice(len(int_to_vocab), 1, p = prob)[0]

    return int_to_vocab[predicted_word]

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)
```

    this
    Tests Passed
    

## Generate TV Script
This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.


```python
gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[0][dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
```

# The TV Script is Nonsensical
It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckily there's more data!  As we mentioned in the beggining of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't have you train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.  After you complete the project, of course.
# Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_tv_script_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
