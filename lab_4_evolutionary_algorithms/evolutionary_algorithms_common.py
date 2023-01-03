def getTutorialDistanceMatrix():
    return [[0, 51.82935435566324, 98.75862089015044, 95.72125776317705, 35.18007760345776, 61.95877979216853],
            [51.82935435566324, 0, 53.64138656826476, 91.7988807780066, 10.360763704960231, 79.64289678096108],
            [98.75862089015044, 53.64138656826476, 0, 26.64406301021866, 47.645192076551, 83.42271036319632],
            [95.72125776317705, 91.7988807780066, 26.64406301021866, 0, 17.925878950612145, 59.723600567316446],
            [35.18007760345776, 10.360763704960231, 47.645192076551, 17.925878950612145, 0, 24.28530285679853],
            [61.95877979216853, 79.64289678096108, 83.42271036319632, 59.723600567316446, 24.28530285679853, 0]]


def generateDistanceMatrix(np, size, lower, upper):
    M = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            p = lower + (upper - lower) * np.random.random()
            M[i][j] = p
            M[j][i] = p
    return M


def displayHistogram(plt, np, matingPool, populationSize):
    D = np.array(matingPool)
    D = D.ravel()
    n, bins, patches = plt.hist(D, populationSize, facecolor='g', alpha=0.75)
    plt.xlabel('Index')
    plt.ylabel('Selected')
    plt.xlim(0, 9)
    # plt.ylim(0, 1.0)
    plt.grid(True)
    plt.show()


def plotConvergence(plt, X, Y_MIN, Y_MEAN, Y_MAX):
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(X, Y_MIN, Y_MAX, alpha=0.5)
    ax.plot(X, Y_MEAN, linestyle='-', color="black")
    plt.xlabel('Generation')
    plt.ylabel('Total distance')
    plt.grid(True)


def getLargeDistanceMatrix():
    return [[0,
             13.11679289207314,
             32.301872570270476,
             80.20074899567066,
             50.31570467317909,
             11.710184729927377,
             31.909721683409558,
             49.551800300375504,
             29.75721212657729,
             58.540222276710125,
             83.69406426261004,
             87.23755830836704,
             33.46391898482703,
             85.0204483205405,
             34.19004950827096],
            [13.11679289207314,
             0,
             26.243403313080577,
             60.947312058736664,
             43.27630082301579,
             37.63783581117576,
             42.28218969909194,
             81.97170499166901,
             77.76989405785204,
             99.43108641064109,
             81.03333898413302,
             60.777029097145906,
             89.91548995258631,
             45.18928464492643,
             19.17126718289063],
            [32.301872570270476,
             26.243403313080577,
             0,
             56.5742074106612,
             29.74240253301107,
             83.82523682253738,
             77.27812879606292,
             19.146602077286342,
             30.499307462678797,
             42.20773683089745,
             18.753119254434544,
             90.53488489549117,
             21.19780076322172,
             37.27010698190553,
             77.3447033522143],
            [80.20074899567066,
             60.947312058736664,
             56.5742074106612,
             0,
             40.0414369373474,
             12.708947664930534,
             53.36803807049654,
             46.28307958869744,
             98.69757579056794,
             90.58491304687855,
             21.671282106795182,
             22.290042365455975,
             31.58408570647778,
             96.87511043731193,
             54.86867271043367],
            [50.31570467317909,
             43.27630082301579,
             29.74240253301107,
             40.0414369373474,
             0,
             37.792667322531614,
             82.17858950419256,
             74.25197587767815,
             93.29071775288023,
             42.57734946343273,
             84.01778301707725,
             38.61664016950756,
             44.190141867195464,
             80.91347448759514,
             19.413195483989522],
            [11.710184729927377,
             37.63783581117576,
             83.82523682253738,
             12.708947664930534,
             37.792667322531614,
             0,
             68.17340720482886,
             60.72978460622417,
             75.89911934573924,
             23.769868177123218,
             27.096043794148567,
             54.465211905279574,
             27.285003892730717,
             37.921754258392625,
             48.815279267750356],
            [31.909721683409558,
             42.28218969909194,
             77.27812879606292,
             53.36803807049654,
             82.17858950419256,
             68.17340720482886,
             0,
             33.64681100297268,
             62.9088569912565,
             88.12518808369917,
             38.01726691465518,
             52.033546550003585,
             41.98079359854138,
             91.54050212167185,
             18.776326746203434],
            [49.551800300375504,
             81.97170499166901,
             19.146602077286342,
             46.28307958869744,
             74.25197587767815,
             60.72978460622417,
             33.64681100297268,
             0,
             45.96693454697192,
             48.710853469580336,
             63.6744213522001,
             48.27905724410608,
             73.4065805162746,
             77.17710218395214,
             63.36888547547696],
            [29.75721212657729,
             77.76989405785204,
             30.499307462678797,
             98.69757579056794,
             93.29071775288023,
             75.89911934573924,
             62.9088569912565,
             45.96693454697192,
             0,
             34.172830999459805,
             36.9519931952971,
             92.65723282606436,
             52.36658601207181,
             78.16954150884014,
             11.256081292416164],
            [58.540222276710125,
             99.43108641064109,
             42.20773683089745,
             90.58491304687855,
             42.57734946343273,
             23.769868177123218,
             88.12518808369917,
             48.710853469580336,
             34.172830999459805,
             0,
             26.286079590156096,
             13.642288235050186,
             12.404327573554474,
             49.73989457210752,
             39.9785557114806],
            [83.69406426261004,
             81.03333898413302,
             18.753119254434544,
             21.671282106795182,
             84.01778301707725,
             27.096043794148567,
             38.01726691465518,
             63.6744213522001,
             36.9519931952971,
             26.286079590156096,
             0,
             61.365326428101,
             77.41738627746007,
             66.9983312617423,
             67.95974125785867],
            [87.23755830836704,
             60.777029097145906,
             90.53488489549117,
             22.290042365455975,
             38.61664016950756,
             54.465211905279574,
             52.033546550003585,
             48.27905724410608,
             92.65723282606436,
             13.642288235050186,
             61.365326428101,
             0,
             33.34708814715274,
             58.46812801188503,
             62.67270240381819],
            [33.46391898482703,
             89.91548995258631,
             21.19780076322172,
             31.58408570647778,
             44.190141867195464,
             27.285003892730717,
             41.98079359854138,
             73.4065805162746,
             52.36658601207181,
             12.404327573554474,
             77.41738627746007,
             33.34708814715274,
             0,
             10.243660568949362,
             23.643436736212404],
            [85.0204483205405,
             45.18928464492643,
             37.27010698190553,
             96.87511043731193,
             80.91347448759514,
             37.921754258392625,
             91.54050212167185,
             77.17710218395214,
             78.16954150884014,
             49.73989457210752,
             66.9983312617423,
             58.46812801188503,
             10.243660568949362,
             0,
             79.92884754687807],
            [34.19004950827096,
             19.17126718289063,
             77.3447033522143,
             54.86867271043367,
             19.413195483989522,
             48.815279267750356,
             18.776326746203434,
             63.36888547547696,
             11.256081292416164,
             39.9785557114806,
             67.95974125785867,
             62.67270240381819,
             23.643436736212404,
             79.92884754687807,
             0]]
