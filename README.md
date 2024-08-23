# copyright @ Ziming Li
version 1.0
All files are created alone by Ziming Li, a Ph.D student from Tsinghua University, China.
The code is for the final project of my summer internship in Yunzhixin'an Technology Co., LTD, Zhengzhou, China.
If you have any questions, please contact me by email: lzm22@mails.tsinghua.edu.cn

# Problem background:

    在现代信息社会，数据的种类繁多，包括个人信息、财务数据、商业机密、医疗记录等。为了有效管理这些数据，组织通常会根据数据的重要性和敏感性，将其分为几个等级。例如，普通数据、敏感数据和高度敏感数据。普通数据可能是公开的信息，敏感数据则可能涉及个人隐私，而高度敏感数据则可能是涉及国家安全或重大商业利益的信息。
    
    数据分类分级的主要目的在于提高数据管理的效率和安全性。通过对数据进行分类，组织能够更清晰地了解各类数据的特性和风险，从而制定更有针对性的安全策略。例如，对于高度敏感的数据，可能需要实施更严格的访问控制和加密措施，而对于普通数据则可以采取相对宽松的管理策略。此外，数据分类分级还可以帮助组织满足法律法规的要求，确保在数据处理和存储过程中遵循相关的合规标准。
    
    实施数据分类分级的过程通常包括数据识别、分类标准的制定、数据标签的赋予以及定期的审查与更新。通过这些步骤，组织能够持续监控数据的状态，及时调整管理策略，以应对不断变化的安全威胁和业务需求。总之，数据分类分级不仅是信息安全管理的重要组成部分，也是提升组织数据治理能力的有效手段。通过科学合理的数据分类分级，组织能够更好地保护信息资产，降低安全风险，实现可持续发展。
    
    然而，在目前的数据分类分级这一问题中，存在着如下的问题：
    
1.	存在海量的数据标签，人工分类代价巨大。
2.	不同行业或同一行业内部数据标签往往有着大量的重复字段，却未加以利用。
3.	目前依靠计算机进行自动文本识别和打标的方法较为朴素，往往只是通过字符串检测与匹配的方式进行，正确率低，遗漏率高，且可能无法提取出字段的全部特征。

    传统的数据分类和打标的算法主要是依靠以下的想法。由于不同行业，数据表的标签可能是非常海量的，因此无法进行人工的数据分类。所以说，为了对于一个新的数据表中的数据进行分类处理，我们一般采用的方法是通过字符串匹配和字符串识别的方法，对于这一数据表中的数据表头进行分类处理。也就是针对这一数据列表中的表头，它一般是一系列的字符串。通过字符串匹配等方式找到之前已分类评定过的表中与其相似的字段，找到的这些与其相似的字段，就可以完成对于数据分类的划分。然而这种方法其效率较低，而且准确率也比较差，亟待改进。同时，这一方法只利用了文本表头的信息，而没有利用文本中内容的信息。

    目前，人工智能技术飞速发展，成为下一代引领生产力发展的科技动力，我们注意到，以自然语言处理（NLP）为首的一众机器学习与人工智能的算法可以被利用到处理数据分类分级的任务中来。自然语言处理（Natural Language Processing, NLP）是人工智能领域的一个重要分支，旨在使计算机能够理解、解释和生成自然语言。在数据分类分级的背景下，NLP展现出巨大的应用潜力。

    基于这样的背景，我提出了基于文本表头和文本内容的交叉验证的数据分类算法，其基本思想如下。对于文本表头，依然使用与传统方法相似的字符串模糊匹配的算法进行匹配，对于直接能够匹配到相似字段的表头内容，则不进行内容的检测匹配，直接将其划分。对于无法进行匹配的表头内容，我们进入下一阶段的内容匹配。对于表内容进行基于内容的文本分类。表内容是一系列的字符串，基于这些字符串，我们需要利用NLP和神经网络相结合的处理方法，在本项目中，我所用算法如下：

利用自然语言处理的一些Python包对表内容的字符串进行处理，将其转化为一系列的向量，我选用gensim与nltk。

选取神经网络模型，进行分类。由于不同的表内容，其字符串的长度是不一样的，因此为了把所有的字符输入到我们的模型中。我们必须要选用一种能够接受不同长度输入的模型，我选用的是循环神经网络模型

最后，训练神经网络，完成数据分类


# Based on the deep learning method, data classification oriented to data security is realized. 

The functions of the files are described below:

1) data generation: data_gen.ipynb, the generated data is in raw_training_data.csv
2) data pre-processing, turn each word into vectors: pre_process.ipynb
3) data pre-process demo, visualization of the word vectors, showing that similar words have closer word vectors: pre_process_deemo.ipynb
4) training with RNN model with GRU cell (ordinary RNN cell and LSTM cell can also be chosen with a few modifications), after training, the best model is stored in ./trained_model/checkpoint_run.pth: training.ipynb
5) model loading and example showing, the shown examples are stored in ./example_results.csv file: result.ipynb
6) others: model.py and RNNcell.py are important files to define my own RNN network.
7) result example: some examples of predictions are given in example_results.csv
8) the trained model is in ./trained_model
9) data used for training is in ./data: not uploaded cause it's too large

# usage:
first run data_gen.ipynb to generate raw_training_data.csv

then run pre_process.ipynb to get ./data

then run training.ipynb to train model

finally run result.ipynb for some results

# environment:
python==3.8.0

Faker==27.0.0

gensim==4.3.3

numpy==1.24.4

scipy==1.10.1

sklean==1.3.2

torch==2.2.2
