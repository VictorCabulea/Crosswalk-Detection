import cv2
import math
import numpy as np


# Variabila utilizata pentru eliminarea liniilor cu o panta mai mica decat aceasta valoare si mai mare decat (90 - aceasta valoare)
REJECT_DEGREE_TH = 7.0


def filtrareLinii(linii):
    liniiFinale = []

    for linie in linii:
        [[x1, y1, x2, y2]] = linie

        # Calcul ecuatia dreptei: y = mx + c
        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
        else:
            m = 0
        c = y2 - m * x2

        # Theta va contine valori intre -90 si 90
        # atan() - functie care returneaza valoarea arctangentei
        theta = math.degrees(math.atan(m))

        # Eliminarea liniilor care au panta aproape egala cu 0 sau cu 90
        # REJECT_DEGREE_TH = 7.0
        # Sortarea celorlalte linii
        if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):
            # Determinarea lungimii liniilor
            l = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            liniiFinale.append([x1, y1, x2, y2, m, c, l])

    # Se vor pastra doar cele mai lungi 10 linii, restul fiind eliminate pentru a creste vieza programului
    if len(liniiFinale) > 10:
        liniiFinale = sorted(liniiFinale, key=lambda x: x[-1], reverse=True)
        liniiFinale = liniiFinale[:10]

    return liniiFinale


def obtinereLinii(Image):
    # Transformarea imaginii intiale intr-o imagine grayscale
    grayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    # Aplicarea unui filtru Gaussian de 5x5 pentru a elimina zgomotul
    blurGrayImage = cv2.GaussianBlur(grayImage, (5, 5), 1)
    # Generarea contururilor obiectelor
    edgeImage = cv2.Canny(blurGrayImage, 40, 255)

    # Utilizarea transformatei Hough pentru gasirea liniilor
    linii = cv2.HoughLinesP(edgeImage, 1, np.pi / 180, 50, 10, 15)

    # Verificare daca au fost gasite linii sau nu
    if linii is None:
        print("Nu au fost gasite suficiente linii pentru determinarea punctului de disparitie.")
        exit(0)

    # Filtrarea liniilor
    liniiFiltrate = filtrareLinii(linii)

    return liniiFiltrate


def GetVanishingPoint(Lines):
    # Se va utiliza algorimul RANSAC
    # Se vor lua combinari de lnii una cate una si se va gasi puncutl lor de intersectie
    # Se va calcula eroarea de detectarea a punctului de intersecctie ca fiind radical din suma patratelor distantelor de la fiecare linie la punct

    # Algoritmul RANSAC este o tehnica de estimare a parametrilor unui model prin esantionarea aleatoare a datelor.
    # Algorimul RANSAC va filtra in mod optim un set de date care contine atat valori corecte, cat si valori gresite, care trebuie eliminate.

    # Punctul de disparitie - punctul in care liniile din spatiul tridimensional par sa convearga
    VanishingPoint = None

    MinError = 100000000000

    for i in range(len(Lines)):
        for j in range(i + 1, len(Lines)):
            m1, c1 = Lines[i][4], Lines[i][5]
            m2, c2 = Lines[j][4], Lines[j][5]

            if m1 != m2:
                x0 = (c1 - c2) / (m2 - m1)
                y0 = m1 * x0 + c1

                err = 0
                for k in range(len(Lines)):
                    m, c = Lines[k][4], Lines[k][5]
                    m_ = (-1 / m)
                    c_ = y0 - m_ * x0

                    x_ = (c - c_) / (m_ - m)
                    y_ = m_ * x_ + c_

                    l = math.sqrt((y_ - y0) ** 2 + (x_ - x0) ** 2)

                    err += l ** 2 # ** 2 - ridicarea numarului la puterea 2

                err = math.sqrt(err)

                if MinError > err:
                    MinError = err
                    VanishingPoint = [x0, y0]

    return VanishingPoint


if __name__ == "__main__":
    # Legatura camera telefon
    # capture=cv2.VideoCapture("https://192.168.100.24:8080/video")
    # capture = cv2.VideoCapture("https://10.4.195.3:8080/video")

    # Utilizare videoclip
    #capture = cv2.VideoCapture("tokyo.mp4")
    capture = cv2.VideoCapture("road.mp4")


    while (True):
        _, frame = capture.read()

        # doar pentru preluare imagine live de pe camera
        # rotated= cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Decuparea imaginii pentru eliminarea zonelor laterale si a zonei superioare
        # Pentru o mai buna focalizare mai buna pe strada
        #crop = frame[580:1300, 600:1400] # Tokyo
        crop = frame[420:880, 380:1620] # Road

        # Extragerea liniilor din imagine
        Lines = obtinereLinii(crop)

        # Obtinerea punctului de disparitie
        VanishingPoint = GetVanishingPoint(Lines)

        # Verificare existenta punct de disparitie
        if VanishingPoint is None:
            print("Punctul de disparitie nu a fost detectat. Nu sunt suficiente linii gasite in imagine.")
            continue

        # Desenarea liniilor gasite si a punctului de disparitie
        for Line in Lines:
            cv2.line(crop, (Line[0], Line[1]), (Line[2], Line[3]), (0, 255, 0), 10)
        cv2.circle(crop, (int(VanishingPoint[0]), int(VanishingPoint[1])), 10, (0, 0, 255), 18)

        # Afisarea rezultatului
        cv2.imshow('livestream', crop)

        # Pentru inchiderea ferestrei, apasati q sau Q
        if (cv2.waitKey(1) == ord("q") or cv2.waitKey(1) == ord("Q")):
            break

    # Inchiderea ferestrei si eliberarea resurselor uitilizate
    capture.release()
    cv2.destroyAllWindows()