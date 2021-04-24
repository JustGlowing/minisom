<h1>MiniSom<img src='https://3.bp.blogspot.com/-_6UDGEHzIrs/WSfiyjmoeRI/AAAAAAAABHw/3UQylcCBEhUfHNhf56WSHBBmQ6g_lXQhwCLcB/s320/minisom_logo.png' align='right'></h1>

Self Organizing Maps
--------------------

MiniSom is a minimalistic and Numpy based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display. Minisom is designed to allow researchers to easily build on top of it and to give students the ability to quickly grasp its details.

Updates about MiniSom are posted on <a href="https://twitter.com/JustGlowing">Twitter</a>.

Installation
---------------------

Just use pip:

    pip install minisom

or download MiniSom to a directory of your choice and use the setup script:

    git clone https://github.com/JustGlowing/minisom.git
    python setup.py install

How to use it
---------------------

In order to use MiniSom you need your data organized as a Numpy matrix where each row corresponds to an observation or as list of lists like the following:

```python
data = [[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]]      
```

 Then you can train MiniSom just as follows:

```python
from minisom import MiniSom    
som = MiniSom(6, 6, 4, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.train(data, 100) # trains the SOM with 100 iterations
```

You can obtain the position of the winning neuron on the map for a given sample as follows:

```
som.winner(data[0])
```

For an overview of all the features implemented in minisom you can browse the following examples: https://github.com/JustGlowing/minisom/tree/master/examples

#### Export a SOM and load it again

A model can be saved using pickle as follows

```python
import pickle
som = MiniSom(7, 7, 4)

# ...train the som here

# saving the som in the file som.p
with open('som.p', 'wb') as outfile:
    pickle.dump(som, outfile)
```

and can be loaded as follows

```python
with open('som.p', 'rb') as infile:
    som = pickle.load(infile)
```

Note that if a lambda function is used to define the decay factor MiniSom will not be pickable anymore.

Examples
---------------------

Here are some of the charts you'll see how to generate in the examples:

| | |
:-------------------------:|:-------------------------:
Seeds map ![](https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_seed.png)  | Class assignment ![](https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_seed_pies.png)
Handwritteng digits mapping ![](https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_digts.png)  |  Hexagonal Topology <img src="https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_seed_hex.png" alt="som hexagonal toplogy" width=450>
Color quantization ![](https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_color_quantization.png)  |  Outliers detection ![](https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_outliers_detection_circle.png)

Other tutorials
------------
- <a href="https://glowingpython.blogspot.com/2013/09/self-organizing-maps.html">Self Organizing Maps on the Glowing Python</a> by me ;-)
- <a href="http://aa.ssdi.di.fct.unl.pt/files/AA-16_notes.pdf">Lecture notes from the Machine Learning course at the University of Lisbon</a>
- <a href="https://heartbeat.fritz.ai/introduction-to-self-organizing-maps-soms-98e88b568f5d">Introduction to Self-Organizing</a> by Derrick Mwiti
- <a href="http://inphronesys.com/?p=625">Self Organizing Maps on gapminder data</a> [in German]
- <a href="https://medium.com/neuronio/discovering-som-an-unsupervised-neural-network-12e787f38f9">Discovering SOM, an Unsupervised Neural Network</a> by Gisely Alves
- Video tutorials made by the GeoEngineerings School: <a href="https://www.youtube.com/watch?v=3osKNPyAxPM&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC&index=11">Part 1</a>; <a href="https://www.youtube.com/watch?v=uUpQ6MITlVs&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC&index=12">Part 2</a>; <a href="https://www.youtube.com/watch?v=mryFU0TEInk&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC&index=13">Part 3</a>; <a href="https://www.youtube.com/watch?v=9MzFOIoxxdk&index=14&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC">Part 4</a>
- Video tutorial <a href="https://www.youtube.com/watch?v=0qtvb_Nx2tA">Self Organizing Maps: Introduction</a> by SuperDataScience
- <a href="http://docs.unigrafia.fi/publications/kohonen_teuvo/">MATLAB Implementations and Applications of the Self-Organizing Map</a> by Teuvo Kohonen (Inventor of SOM)

How to cite MiniSom
------------
```
@misc{vettigliminisom,
  title={MiniSom: minimalistic and NumPy-based implementation of the Self Organizing Map},
  author={Giuseppe Vettigli},
  year={2018},
  url={https://github.com/JustGlowing/minisom/},
}
```

Who uses Minisom?
------------

<ul>
<li>
Gorgoglione, Angela, Alberto Castro, Vito Iacobellis, and Andrea Gioia. <a href="https://www.mdpi.com/2071-1050/13/4/2054/pdf">A Comparison of Linear and Non-linear Machine Learning Techniques (PCA and SOM) for Characterizing Urban Nutrient Runoff</a>. Sustainability 13, no. 4. 2021.
</li>
<li>
Mazin A, Hawkins SH, Stringfield O, Dhillon J, Manley BJ, Jeong DK, Raghunand N. <a href="https://www.nature.com/articles/s41598-021-83271-4">Identification of sarcomatoid differentiation in renal cell carcinoma by machine learning on multiparametric MRI</a>. Nature, Scientific Reports. 2021.
</li>
<li>
Qi J, Ma G, Navarro-Alarcon D, Zhang H, Lyu Y. <a href="https://arxiv.org/pdf/2101.07513.pdf">Towards Latent Space Based Manipulation of Elastic Rods using Autoencoder Models and Robust Centerline Extractions</a>. arXiv:2101.07513. 2021.
</li>
<li>
Julianna C. Oliveira, Eduardo Zorita, Vimal Koul, Thomas Ludwig, Johanna Baehr.<a href="https://dl.acm.org/doi/abs/10.1145/3429309.3429319">Forecast opportunities for European summer climate ensemble predictions using Self-Organising Maps</a>. CI2020: Proceedings of the 10th International Conference on Climate Informatics. 2020.
</li>
<li>
Gorgoglione, A., Castro, A., Gioia, A., & Iacobellis, V. <a href="https://link.springer.com/chapter/10.1007/978-3-030-58811-3_49">Application of the Self-organizing Map (SOM) to Characterize Nutrient Urban Runoff</a>. International Conference on Computational Science and Its Applications. Springer, Cham, 2020.
</li>        
<li>
Bonelli Toro, A. G., and M. P. Gómez. <a href="https://www.researchgate.net/profile/Miguel_Zitto/publication/344397710_Energy_and_Shannon_entropy_in_acoustic_emission_signals_for_damage_evaluation_in_andesite_rock/links/5f70a7f7458515b7cf511057/Energy-and-Shannon-entropy-in-acoustic-emission-signals-for-damage-evaluation-in-andesite-rock.pdf#page=123">Machine Learning Applied to Acoustic Emission tor Tool Wear Classification during Milling of Composite Materials</a>. An International Forum For The AE Science and Technology. Vol. 37. 2020.
</li>    
<li>
Mancini, R., Ritacco, A., Lanciano, G., & Cucinotta, T. <a href="http://retis.sssup.it/~tommaso/publications/IEEE-SBAC-PAD-2020.pdf">XPySom: High-Performance Self-Organizing Maps</a>. Proceedings of IEEE 32nd International Symposium on Computer Architecture and High Performance Computing. 2020.
</li>
<li>
Chen, Yang, Nami Ashizawa, Seanglidet Yean, Chai Kiat Yeo, and Naoto Yanai. <a href="https://arxiv.org/pdf/2008.12686.pdf">Self-Organizing Map assisted Deep Autoencoding Gaussian Mixture Model for Intrusion Detection</a>. arXiv preprint arXiv:2008.12686 2020.
</li>
<li>
Athanasakis E, <a href="http://ikee.lib.auth.gr/record/320630/files/GRI-2020-28133.pdf">Data-Analysis in environmental and traffic data for Thessaloniki Greece</a>. Master Thesis at Aristotle University of Thessaloniki. 2020.
</li>
<li>
Schillaci G, Ciria A, Lara B. <a href="https://arxiv.org/pdf/2007.14632.pdf">Tracking Emotions: Intrinsic Motivation Grounded on Multi-Level Prediction Error Dynamics</a>. IEEE ICDL-Epirob 2020. 2020.
</li>
<li>
Massaro, Alessandro, Giuseppe Mastandrea, Luigi D'Oriano, Giuseppe Rocco Rana, Nicola Savino, and Angelo Galiano. <a href="https://ieeexplore.ieee.org/abstract/document/9138231">Systems for an intelligent application of Automated Processes in industry: a case study from “PMI IoT Industry 4.0” project</a>. 2020 IEEE International Workshop on Metrology for Industry 4.0 & IoT. 2020.
</li>
<li>
Ko, Ili, Desmond Chambers, and Enda Barrett. <a href="https://arxiv.org/pdf/2003.10454">Feature dynamic deep learning approach for DDoS mitigation within the ISP domain</a>. International Journal of Information Security. 2020.
</li>
<li>
Wesam Salah Alaloul and Abdul Hannan Qureshi. <a href="https://www.intechopen.com/online-first/data-processing-using-artificial-neural-networks">MData Processing Using Artificial Neural Networks</a>. In Dynamic Data Assimilation-Beating the Uncertainties. IntechOpen, 2020.
</li>
<li>
Gennadi Lessin, Luca Polimene, Yuri Artioli, Momme Butenschön, Darren R. Clark, Ian Brown, Andrew P. Rees. <a href="https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JG005613">Modeling the seasonality and controls of nitrous oxide emissions on the northwest European continental shelf</a>. Journal of Geophysical Research: Biogeosciences. 2020.
</li>
<li>
Jorge Amaya, Romain Dupuis, Maria Elena Innocenti, and Giovanni Lapenta. <a href="https://arxiv.org/pdf/2004.13430.pdf">Visualizing and Interpreting Unsupervised Solar Wind Classifications</a>. Frontiers in Astronomy and Space Sciences Space Physics. 2020.
</li>
<li>
Sandipan Dey. <a href="https://www.amazon.com/Python-Image-Processing-Cookbook-processing-ebook/dp/B084ZN7Y5F">Python Image Processing Cookbook: Over 60 recipes to help you perform complex image processing and computer vision tasks with ease</a>. Packt Publishing Ltd, April 2020.
</li>
<li>
Odestål, Oscar and Palmqvist Sjövall, Anna. <a href="https://lup.lub.lu.se/student-papers/search/publication/9007379">Adaptive Reference Images for Blood Cells using Variational Autoencoders and Self-Organizing Maps</a>. Master Thesis, Lund University. 2020.
</li>
<li>
Hadleigh D. Thompson  Stephen J. Déry  Peter L. Jackson  Bernard E. Laval. <a href="https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/joc.6560">A synoptic climatology of potential seiche‐inducing winds in a large intermontane lake: Quesnel Lake, British Columbia, Canada</a>. International Journal of Climatology. 2020.
</li>
<li>
Benyamin Motevalli, Baichuan Sun, Amanda S. Barnard. <a href="https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.9b10615">Understanding and Predicting the Cause of Defects in Graphene Oxide Nanostructures Using Machine Learning</a>. The Journal of Physical Chemistry C. 2020.
</li>
<li>
Daniel L. Donaldson, Dilan Jayaweera. <a href="https://www.sciencedirect.com/science/article/abs/pii/S0142061519322859">Effective solar prosumer identification using net smart meter data</a>. International Journal of Electrical Power & Energy Systems
Volume 118. 2020.
</li>
<li>
Pauli Tikka, Moritz Mercker, Ilya Skovorodkin, Ulla Saarela, Seppo Vainio, Veli-Pekka Ronkainen, James P. Sluka, James A. Glazier, Anna Marciniak-Czochra, Franz Schaefer. <a href="https://www.biorxiv.org/content/biorxiv/early/2020/01/14/2020.01.14.905711.full.pdf">Computational Modelling of Nephron Progenitor Cell Movement and Aggregation during Kidney Organogenesis</a>. Pre-print on biorxiv.org. 2020.
</li>    
<li>
Felix M. Riese, Sina Keller, Stefan Hinz. <a href="https://www.mdpi.com/2072-4292/12/1/7/pdf">Supervised and Semi-Supervised Self-Organizing Maps for Regression and Classification Focusing on Hyperspectral Data</a>. Remote Sensing, special Issue Advanced Machine Learning Approaches for Hyperspectral Data Analysis. 2020.
</li>
<li>
Giobergia, Flavio, and Elena Baralis. <a href="https://ieeexplore.ieee.org/abstract/document/9006055">Fast Self-Organizing Maps Training</a>. 2019 IEEE International Conference on Big Data (Big Data). IEEE, 2019.
</li>
<li>
Silva, Roberto F., Gustavo M. Mostaço, Fernando Xavier, Antonio Mauro Saraiva, and Carlos E. Cugnasca. <a href="https://efita-org.eu/wp-content/uploads/2020/02/24.-efita7.pdf">COMPARISON OF THE K-MEANS AND SELF-ORGANIZING MAPS TECHNIQUES TO LABEL AGRICULTURAL SUPPLY CHAIN DATA</a>. Digitizing Agriculture, 12th EFITA International Conference. 2019.
</li>
<li>
Üstünkök, Tolga, Ozan Can Acar, and Murat Karakaya. <a href="https://ieeexplore.ieee.org/abstract/document/8965477/">Image Tag Refinement with Self Organizing Maps.</a>. 2019 1st International Informatics and Software Engineering Conference (UBMYK). IEEE, 2019.
</li>
<li>
Rohana, N. A., Yusof, N., Uti, M. N., and Din, A. H. M. <a href="https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-4-W16/543/2019/">Exploring spatio-temporal wave pattern using unsupervised technique</a>. Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLII-4/W16, 543–548. 2019.    
</li>
<li>
Nguyen, Thanh Hai. <a href="https://link.springer.com/chapter/10.1007/978-3-030-35653-8_20">Metagenome-Based Disease Classification with Deep Learning and Visualizations Based on Self-organizing Maps</a>. International Conference on Future Data and Security Engineering. Springer, Cham, 2019.
</li>
<li>
Ujjawal Kamal Panchal, Sanjay Verma. <a href="https://ieeexplore.ieee.org/abstract/document/8944605">Identification of Potential Future Credit Card Defaulters from Non Defaulters using Self Organizing Maps</a>. 10th International Conference on Computing, Communication and Networking Technologies (ICCCNT). 2019.    
</li>
<li>
Mengxia Luo, Can Yang, Xiaorui Gong, Lei Yu. <a href="https://link.springer.com/chapter/10.1007/978-3-030-37228-6_16">FuncNet: A Euclidean Embedding Approach for Lightweight Cross-platform Binary Recognition</a>. Security and Privacy in Communication Networks, 15th EAI International Conference, SecureComm 2019.    
</li>
<li>
Melvin Gelbard. <a href="https://project-archive.inf.ed.ac.uk/msc/20193475/msc_proj.pdf">A Data Mining Approach to the Study of Dynamic Changes in Brain White Matter</a>. Master Thesis, University of Edinburgh. 2019.    
</li>
<li>
E Mele, C Elias, A Ktena. <a href="https://content.sciendo.com/configurable/contentpage/journals$002fecce$002f15$002f1$002farticle-p21.xml">Machine Learning Platform for Profiling and Forecasting at Microgrid Level</a>. Electrical, Control and Communication Engineering. 2019.    
</li>
<li>
György Kovács. <a href="https://www.sciencedirect.com/science/article/pii/S0925231219311622">Smote-variants: A python implementation of 85 minority oversampling techniques</a>. Neurocomputing, 2019 - Elsevier.
</li>
<li>
Catalin Stoean, Ruxandra Stoean, Roberto Antonio Becerra-García, Rodolfo García-Bermúdez, Miguel Atencia, Francisco García-Lagos, Luis Velázquez-Pérez, Gonzalo Joya. <a href="https://link.springer.com/chapter/10.1007/978-3-030-20518-8_3">Unsupervised Learning as a Complement to Convolutional Neural Network Classification in the Analysis of Saccadic Eye Movement in Spino-Cerebellar Ataxia Type 2</a>. IWANN 2019: Advances in Computational Intelligence pp 26-37, 2019.
</li>
<li>
I Ko, D Chambers, E Barrett . <a href="http://ceur-ws.org/Vol-2421/MEX-A3T_paper_8.pdf">Feature dynamic deep learning approach for DDoS mitigation within the ISP domain</a>. International Journal of Information Security, 2019.
</li>
<li>
Leonardo Barreto, Edjard Mota. <a href="https://arxiv.org/abs/1906.06761">Self-organized inductive reasoning with NeMuS</a>. June 2019.
</li>
<li>
Casavantes, Marco, Roberto López, Luis Carlos González-Gurrola, and Manuel Montes-y-Gómez. <a href="http://ceur-ws.org/Vol-2517/T3-16.pdf">UACh-INAOE at HASOC 2019: Detecting Aggressive Tweets by Incorporating Authors' Traits as Descriptors</a>. In FIRE (Working Notes). 2019.
</li>
<li>
Marco Casavantes, Roberto Lopez, and Luis Carlos Gonzalez. <a href="http://ceur-ws.org/Vol-2421/MEX-A3T_paper_8.pdf">UACh at MEX-A3T 2019: Preliminary Results on Detecting Aggressive Tweets by Adding Author Information Via an Unsupervised Strategy</a>. Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2019), 2019.
</li>
<li>
G Roma, O Green, PA Tremblay. <a href="https://pure.hud.ac.uk/ws/files/17022863/adaptive_mapping.pdf">Adaptive Mapping of Sound Collections for Data-driven Musical Interfaces</a>. 19th edition of NIME, 2019.
</li>
<li>
H. D. Thompson. Wind climatology of Quesnel Lake, British Columbia. Master Thesis, University of Northern British Columbia, Prince George, BC. 2019.
</li>
<li>
Florent Forest, Mustapha Lebbah, Hanene Azzag and Jérôme Lacaille. <a href="http://florentfo.rest/files/LDRC-2019-DeepArchitecturesJointClusteringVisualization-full-paper.pdf">Deep Architectures for Joint Clustering and Visualization with Self-Organizing Maps</a>. LDRC@PAKDD 2019 (Learning Data Representation for Clustering@PAKDD) Macau China. 2019.
</li>
<li>
Stephanie Kas. <a href="http://fb07-indico.physik.uni-giessen.de:8080/wiki/images/e/ea/Bachelor_Thesis_Stephanie_Kaes_2019-cmprd.pdf">Multiparameter Analysis of the Belle II Pixeldetector’s Data</a>. Bachelor Thesis, University of Giessen, July 2019.
</li>
<li>
Katharina Dort. <a href="https://docs.belle2.org/record/1382/files/BELLE2-MTHESIS-2019-003.pdf">Search for Highly Ionizing Particles with the Pixel Detector in the Belle II Experiment</a>. Master Thesis, University of Giessen, May 2019.
</li>
<li>
Rahul Kumar. <a href="https://www.amazon.co.uk/Machine-Learning-Quick-Reference-Essential/dp/1788830571">Machine Learning Quick Reference: Quick and essential machine learning hacks for training smart data models</a>. Packt Publishing Ltd, 31 Jan 2019.
</li>
<li>
Michaela Vystrčilova. <a href="https://dspace.cuni.cz/bitstream/handle/20.500.11956/108348/130257629.pdf?sequence=1">Similarity methods for music recommender systems</a>. Bachelor Thesis in Computer Science, Charles University, 2019.
</li>
<li>
Felix M. Riese, Sina Keller. <a href="https://arxiv.org/pdf/1903.11114.pdf">SUSI: Supervised Self-Organizing Maps for Regression and Classification in Python</a>.
</li>
<li>
Dogo, E. M., et al. <a href="https://ieeexplore.ieee.org/abstract/document/8769276/">Sensed Outlier Detection for Water Monitoring Data and a Comparative Analysis of Quantization Error Using Kohonen Self-Organizing Maps</a>. 2018 International Conference on Computational Techniques, Electronics and Mechanical Systems (CTEMS). IEEE, 2018.
</li>
<li>
Y. Xie, L.Le, Y. Zhou, V. V. Raghavan. <a href="https://books.google.co.uk/books?id=gRJrDwAAQBAJ&pg">Deep Learning for Natural Language Processing</a>. Chapter of Computational Analysis and Understanding Natural Languages, Elsevier, 2018
</li>
<li>
Enea Mele, Charalambos Elias, Aphrodite Ktena. <a href="https://ieeexplore.ieee.org/abstract/document/8659866">Electricity use profiling and forecasting at microgrid level</a>. IEEE 59th International Scientific Conference on Power and Electrical Engineering of Riga Technical University (RTUCON), 2018.
</li>
<li>
Chintan Shah, Anjali Jivani. <a href="https://ieeexplore.ieee.org/abstract/document/8554848">A Hybrid Approach of Text Summarization Using Latent Semantic Analysis and Deep Learning</a>. 2018 International Conference on Advances in Computing, Communications and Informatics (ICACCI), 2018.
</li>
<li>
Katsutoshi Masai. <a href="http://koara.lib.keio.ac.jp/xoonips/modules/xoonips/download.php/KO50002002-20184959-0003.pdf?file_id=137977">Facial Expression Classification Using Photo-reflective Sensors on Smart Eyewear</a>. Keio University, Doctoral Thesis, 2018.
</li>
<li>
Katsutoshi Masai, Kai Kunze, Yuta Sugiura, Maki Sugimoto. <a href="https://dl.acm.org/citation.cfm?id=3267562">Mapping Natural Facial Expressions Using Unsupervised Learning and Optical Sensors on Smart Eyewear</a>. Proceedings of the 2018 ACM International Joint Conference and 2018 International Symposium on Pervasive and Ubiquitous Computing and Wearable Computers, 2018 ACM.
</li>
<li>
Ili Ko, Desmond Chambers, Enda Barrett. <a href="https://ieeexplore.ieee.org/document/8343119/">A Lightweight DDoS Attack Mitigation System within the ISP Domain Utilising Self-organizing Map</a>. Proceedings of the Future Technologies, 2018 Springer.
</li>
<li>
T. M. Nam et al. <a href="https://ieeexplore.ieee.org/document/8343119/">Self-organizing map-based approaches in DDoS flooding detection using SDN</a>. 2018 International Conference on Information Networking (ICOIN), 2018.
</li>
<li>
Li Yuan <a href="http://digitalcommons.uri.edu/theses/1244/">Implementation of Self-Organizing Maps with Python</a>. Master Thesis, University of Rhode Island, 2018.
</li>
<li>
Ying Xie, Linh Le, Yiyun Zhou, Vijay V.Raghavan. <a href="https://www.sciencedirect.com/science/article/pii/S0169716118300026">Deep Learning for Natural Language Processing</a>. Elsevier Handbook of Statistics, 2018.
</li>
<li>
André de Vasconcelos Santos Silva. <a href="https://repositorio.iscte-iul.pt/bitstream/10071/18245/1/Master_Andre_Santos_Silva.pdf">Sparse Distributed Representations as Word
Embeddings for Language Understanding</a>. Master Thesis, University Institute of Lisbon, 2018.
</li>
<li>
Vincent Fortuin, Matthias Hüser, Francesco Locatello, Heiko Strathmann, and Gunnar Rätsch. <a href="https://arxiv.org/pdf/1806.02199.pdf">Deep Self-Organization: Interpretable Discrete
Representation Learning on Time Series</a>. 2018.
</li>
<li>
John Mwangi Wandeto. <a href="https://publication-theses.unistra.fr/public/theses_doctorat/2018/Wandeto_John_2018_ED269.pdf">Self-Organizing Map Quantization Error Approach for Detecting Temporal Variations in Image Sets</a>. Doctoral Thesis, University of Strasbourg, 2018.
</li>    
<li>
Birgitta Dresp-Langley, John Mwangi Wandeto, Henry Okola Nyongesa. <a href="https://www.openscience.fr/IMG/pdf/iste_fromd2d17v1n2.pdf">Using the quantization error from Self‐Organizing Map (SOM) output for fast detection of critical variations in image time series</a>. ISTE OpenScience, 2018.
</li>
<li>
John M. Wandeto, Henry O. Nyongesa, Birgitta Dresp-Langley. <a href="https://arxiv.org/abs/1803.11125">Detection of Structural Change in Geographic Regions of Interest by Self Organized Mapping: Las Vegas City and Lake Mead across the Years</a>. 2018.
</li>
<li>
Denis Mayr Lima Martins, Gottfried Vossen, Fernando Buarque de Lima Neto. <a href="https://ieeexplore.ieee.org/abstract/document/8285698/">Learning database queries via intelligent semiotic machines</a>. IEEE Latin American Conference on Computational Intelligence (LA-CCI), 2017.
</li>
<li>
Udemy online course. <a href="https://www.udemy.com/deeplearning/?utm_campaign=email&utm_source=sendgrid.com&utm_medium=email">Deep Learning A-Z™: Hands-On Artificial Neural Networks</a>
<img src="https://www.udemy.com/staticx/udemy/images/v6/logo-coral.svg" height="20">
</li>
<li>
Fredrik Broch Elgaaen, Nicholas Mowatt Larssen. <a href="https://www.duo.uio.no/bitstream/handle/10852/57507/master_nml.pdf?sequence=1&isAllowed=y">Data mining i banksektoren - Prediksjonsmodellering og analyse av kunder som sier opp boliglån</a>. University of Oslo, May 2017.
</li>
<li>
Óscar Clavería González, Enric Monte Moreno, Salvador Torra Porras. <a href="http://www.sciencedirect.com/science/article/pii/S2110701715000694">A self-organizing map analysis of survey-based agents׳ expectations before impending shocks for model selection: The case of the 2008 financial crisis</a>. International Economics Volume 146, Pages 40–58. August 2016.
</li>
<li>
Sameen Mansha, Faisal Kamiran, Asim Karim, Aizaz Anwar. <a href="http://link.springer.com/chapter/10.1007/978-3-319-46681-1_16">A Self-Organizing Map for Identifying InfluentialCommunities in Speech-based Networks</a>. Proceeding CIKM '16 Proceedings of the 25th ACM International on Conference on Information and Knowledge Management Pages 1965-1968. 2016.
</li>
<li>
Sameen Mansha, Zaheer Babar, Faisal Kamiran, Asim Karim. <a href="http://link.springer.com/chapter/10.1007/978-3-319-46681-1_16">Neural Network Based Association Rule Mining from Uncertain Data</a>. Neural Information Processing Volume 9950 of the series Lecture Notes in Computer Science pp 129-136. 2016.
</li>
<li>
Makiyama, Vitor Hirota, M. Jordan Raddick, and Rafael DC Santos. <a href="http://ceur-ws.org/Vol-1478/paper7.pdf">Text Mining Applied to SQL Queries: A Case Study for the SDSS SkyServer</a>. 2nd Annual International Symposium on Information Management and Big Data. 2015.
</li>
<li>
Remi Domingues. <a href="http://www.diva-portal.org/smash/get/diva2:897808/FULLTEXT01.pdf">Machine Learning for
Unsupervised Fraud Detection</a>. Royal Institute of Technology School of Computer Science and Communication KTH CSC. 2015.
</li>
<li>Ivana Kajić, Guido Schillaci, Saša Bodiroža, Verena V. Hafner, <a href="http://dl.acm.org/citation.cfm?id=2559816">Learning hand-eye coordination for a humanoid robot using SOMs</a>. Proceedings of the 2014 ACM/IEEE international conference on Human-robot interaction
Pages 192-193.</li>
</ul>

Guidelines to contribute
---------------------
1. In the description of your Pull Request explain clearly what does it implements/fixes and your changes. Possibly give an example in the description of the PR. In cases that the PR is about a code speedup, report a reproducible example and quantify the speedup.
2. Give your pull request a helpful title that summarises what your contribution does. 
3. Write unit tests for your code and make sure the existing tests are up to date. `pytest` can be used for this:
```
pytest minisom.py
```
4. Make sure that there a no stylistic issues using `pycodestyle`:
```
pycodestyle minisom.py
```
5. Make sure your code is properly commented and documented. Each public method needs to be documented as the existing ones.
