# FACT

To run the colab notebook you need to add in your own github api key to allow the notebook to download the rest of the github.

### Open in Google Colab 🚀

If this is a private repository, follow these steps:

1. Open [Google Colab](https://colab.research.google.com).
2. Click the "GitHub" tab.
3. Sign in with your GitHub account.
4. Search for this repository and open the notebook manually. (open experiment for most experiments, figure3 for figure 3 and figure5 for figure 5)
5. Set runtime to T4 GPU
6. Add your personal acces token in the notebook. (this is needed to download the github repo into colab)
7. The code is pre run but if you want to run things the code needs to be run sequentially else it might not work.


Figure 4 has been made by running the keyword extraction code. AFter this the captions are filtered to the captions containing one of the keywords and has a missclasification. The keyword used for search are picked from the biased keywords (positive clip score).

The label diagnosis was done by running the LabelDiagnoser code and than manually looking through the results in the csv to find if any of them are misslabeled or not. 

For the debias code see the readme in the b2t/b2t_debias