import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
np.set_printoptions(threshold=sys.maxsize)


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)
connection = sqlite3.connect("/home/geekboyboss/Desktop/PCA_from_scratch/PCA_3D_analysis/data.db")
cur = connection.cursor()
cur.execute("SELECT budget, popularity, runtime, revenue, vote_average, vote_count, genre_name, title FROM movies JOIN movie_genres ON movie_genres.movie_id = movies.movie_id JOIN genre ON genre.genre_id = movie_genres.genre_id WHERE vote_count>=50 AND budget>999 AND runtime>0 AND revenue>999 GROUP BY title;")
fetch = cur.fetchall()
connection.close()

features = ["budget", "popularity", "runtime", "revenue", "vote_average", "vote_count", "genre", "title"]
df = pd.DataFrame(data=fetch, columns=features)
print(df)




show_all_movies_catégories_in_graphs = False
show_sphere_of_correlations=True
show_info_about_the_best_movies=True





x = df.loc[:, features[:-2]].values  # Separating out the features
X = np.array(x)
n = X.shape[0]
print("n : {}".format(n))
y = df.loc[:, ['genre']].values  # Separating out the target
titres = list(df.loc[:, ['title']].values)
x = StandardScaler().fit_transform(x)  # Standardiser
nb_composantes = 3  # 3D
pca = PCA(n_components=nb_composantes)  #
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2', 'principal component 3'][:nb_composantes])

# print(np.corrcoef(x))
finalDf = pd.concat([principalDf, df[['genre']]], axis=1)


# Preparing colors for the graphs
targets = ['Adventure', 'Fantasy', 'Animation', 'Drama', 'Horror', 'Action', 'Comedy', 'History', 'Western', 'Thriller', 'Crime', 'Documentary', 'Science Fiction', 'Mystery', 'Music', 'Romance', 'Family', 'War', 'Foreign', 'TV Movie']
NUM_COLORS = len(targets)
cm = plt.get_cmap('gist_rainbow')


if(show_all_movies_catégories_in_graphs==True):
  for i in range(len(targets)):
      fig = plt.figure(figsize=(8, 8))
      ax = plt.axes(projection='3d')
      # ax = fig.add_subplot(1,1,1)
      ax.set_xlabel('Principal Component 1', fontsize=15)
      ax.set_ylabel('Principal Component 2', fontsize=15)
      ax.set_zlabel('Principal Component 3', fontsize=15)

      ax.set_title('{} component PCA'.format(nb_composantes))  # , targets[i]), fontsize = 20)
      colors = ['k' for _ in range(NUM_COLORS)]
      colors[i] = 'b'  # cm(1.*i/NUM_COLORS)
      # colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
      k = 0
      for target, color in zip(targets, colors):
          indicesToKeep = finalDf['genre'] == target
          ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                    finalDf.loc[indicesToKeep, 'principal component 2'],
                    finalDf.loc[indicesToKeep, 'principal component 3'],
                    c=color,
                    s=50,
                    alpha=.05+.95*int(k == i))
          if k == i:
              j = -1
              indice = np.where(indicesToKeep)[0]
              for x, y, z in zip(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], finalDf.loc[indicesToKeep, 'principal component 3']):
                  j += 1
                  if j % (1+len(indice)//10):
                      continue
                  ax.text(x,
                          y,
                          z,
                          titres[indice[j]][0],
                          color='b')
          k += 1
      ax.legend(targets)
      ax.grid()
      #plt.axis('equal')

print("Valeurs propres :",pca.explained_variance_)  # Valeurs propres
print(pca.explained_variance_ratio_)  # Inerties
# Plot a variable factor map for the first two dimensions. (Cercle des corrélations)




if show_sphere_of_correlations==True:
  plt.figure(21)
  ax = plt.axes(projection='3d')
  for i in range(0, len(features)-2):  # len(pca.components_)):
      ax.arrow3D(0, 0, 0, # Start the arrow at the origin
              pca.components_[0, i], pca.components_[1, i], pca.components_[2, i]) # 0 and 1 correspond to dimension 1 and 2
              # head_width=0.1, head_length=0.1)
      ax.text(pca.components_[0, i] + 0.05, pca.components_[1, i] + 0.05, pca.components_[2, i] + 0.05, df.columns.values[i])

  an = np.linspace(0, 2 * np.pi, 100)  # Add a unit circle for scale
  ax.plot3D(np.zeros(an.shape[0]), np.cos(an), np.sin(an))
  ax.plot3D(np.cos(an), np.zeros(an.shape[0]), np.sin(an))
  ax.plot3D(np.cos(an), np.sin(an), np.zeros(an.shape[0]))
  ax.set_xlabel('Principal Comp 1', fontsize=15)
  ax.set_ylabel('Principal Comp 2', fontsize=15)
  ax.set_zlabel('Principal Comp 3', fontsize=15)
  #ax.axis('equal')
  ax.set_title('Variable factor map (Sphere of correlations)')



if show_info_about_the_best_movies==True:
  X_ = (X-(np.sum(X, axis=0)/n))/np.sqrt(np.sum((X-np.sum(X, axis=0)/n)**2, axis=0)/n)
  mu = np.sum(X, axis=0)/n
  sigma = np.sqrt(np.sum((X-(np.sum(X, axis=0)/n))**2, axis=0)/n)
  C = np.dot(X_.T, X_)/n
  D, V = np.linalg.eig(C)
  a = np.flip(np.argsort(D))
  I = D/np.sum(D)
  I = np.flip(np.sort(I))
  I1 = np.cumsum(I)
  location = np.where(I1 >= 0.9)
  S = np.dot(X_, V)

  ctr = (S**2)/np.sum(S**2, axis=0)
  Q = (S**2)/np.reshape(np.sum(S**2, axis=1), (1, n)).T
  pays_cont = []
  seuil = 0.01
  # seuil = 0.00
  print(n)
  if nb_composantes == 2:
      for i in range(n):
          if ctr[i, a[0]] > seuil or ctr[i, a[1]] > seuil:
              pays_cont.append(i)
              print("\n{} : {}".format(titres[i][0], i))
              print("Contributions : ", ctr[i, [a[0], a[1]]])
              print("Qualités :", Q[i, [a[0], a[1]]])
  elif nb_composantes == 3:
      for i in range(n):
          if ctr[i, a[0]] > seuil or ctr[i, a[1]] > seuil or ctr[i, a[2]] > seuil:
              pays_cont.append(i)
              print("\n{} : {}".format(titres[i][0], i))
              print("Contributions :", ctr[i, [a[0], a[1], a[2]]])
              print("Qualité :", Q[i, [a[0], a[1], a[2]]])
  print(pays_cont)
  print("\nLe nombre de films à contribution sup à {} est {}".format(seuil, len(pays_cont)))

  """
  for i in range(0,n,20):
      ax.annotate(titres[i][0], (S[i,a[0]], S[i,a[1]],S[i,a[0]]))
      ax.legend(labels_region, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
  """
plt.show()
