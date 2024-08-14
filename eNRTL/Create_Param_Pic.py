import pandas as pd
import os
from pdf2image import convert_from_path
from tably import get_latex
from pdflatex import PDFLaTeX
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def create_param_pic(df, title):

    molecules = {}
    molecules['H2O'] = 'H_2O'
    molecules['MEA'] = 'MEA'
    molecules['CO2'] = 'CO_2'
    eq_constant = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}

    # Converts dataframe to a latex supported dataframe
    df['Uncertainty'] = df['Uncertainty'].astype(str)

    for i, row in df.iterrows():
        name = row['Name']
        if name[7] == 'L':
            if name[17] != "\'":
                species = name[17:-1].split(',')
                molecule, ions = species[0], species[1:]
                ion_1, ion_2 = ions[0][1:], ions[1][1:-1]
                beg = '$' + name[15] + '_{'
                end = ')}$'
                together = beg + molecules[molecule] + ',(' + ion_1 + ', ' + ion_2 + end
                df.loc[i, 'Description'] = 'eNRTL m-ca'
                df.loc[i, 'Name'] = together
            else:
                species = name[17:-1].split(',')
                ions, molecule = species[:-1], species[-1]
                ion_1, ion_2 = ions[0][1:], ions[1][1:-1]
                beg = '$' + name[15] + '_{'
                end = '}$'
                together = beg + '(' + ion_1 + ', ' + ion_2 + '), ' + molecules[molecule] + end
                df.loc[i, 'Description'] = 'eNRTL ca-m'
                df.loc[i, 'Name'] = together
        else:
            reaction = name.split('_')[2]
            df.loc[i, 'Description'] = reaction.capitalize() + ' rxn Eq'
            df.loc[i, 'Name'] = eq_constant[name[-1]]

        df.loc[i, 'Uncertainty'] = '$\pm$' + ' ' + row["Uncertainty"]

    df['Value'] = df['Value'].map('{:.1f}'.format)

    # df.sort_values(by=['Name'], ascending=True, inplace=True)
    df.sort_values(by=['Percent'], ascending=True, inplace=True)
    df['Percent'] = df['Percent'].map('{:.0%}'.format)

    os.chdir(r"C:\Users\Tanner\Documents\git\MEA\data\Parameters")
    df.to_csv('Parameters.csv', index=False)

    # Gets a string of latex code that creates a table from a csv file
    latex = get_latex(['Parameters.csv'], title)

    # Generates a latex file with the string of latex code
    with open("Parameters.tex", "w") as file:
        file.write(latex)

    # Runs the latex file and writes the bytes output to a pdf file
    pdfl = PDFLaTeX.from_texfile("Parameters.tex")
    pdfl.set_interaction_mode()
    pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False, )
    with open('Parameters.pdf', 'wb') as f:
        f.write(pdf)

    # Converts the pdf into an image
    image = convert_from_path('Parameters.pdf', dpi=200)
    os.remove('Parameters.pdf')

    # Saves the image object as an image file
    image[0].save('Parameters.png')
    print('Updated Image')
    # os.remove('Parameters.tex')


if __name__ == '__main__':
    df = pd.read_csv('../data/Parameters/ParametersOG.csv')
    create_param_pic(df, 'Testing title')
    img = np.asarray(Image.open(r'C:\Users\Tanner\Documents\git\MEA\data\Parameters\Parameters.png'))
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

