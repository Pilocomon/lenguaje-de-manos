import cv2
import mediapipe as mp
import math
from collections import deque

# ============================================================
# FUNCIONES GENERALES
# ============================================================

def distancia(p1, p2):
    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
    )


def draw_bounding_box(image, hand_landmarks):

    h, w, _ = image.shape

    xs = []
    ys = []

    for landmark in hand_landmarks.landmark:
        xs.append(int(landmark.x * w))
        ys.append(int(landmark.y * h))

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    cv2.rectangle(
        image,
        (x_min - 20, y_min - 20),
        (x_max + 20, y_max + 20),
        (0, 255, 0),
        2
    )


def punto(hand, numero, ancho, alto):
    return (
        int(hand.landmark[numero].x * ancho),
        int(hand.landmark[numero].y * alto)
    )


# ============================================================
# DETECCIÓN DE DEDOS
# ============================================================

def dedo_arriba(tip, pip):
    """
    En OpenCV:
    menor Y = más arriba en la imagen.
    """
    return tip[1] < pip[1]


def dedo_abajo(tip, pip):
    return tip[1] > pip[1]


# ============================================================
# DETECTAR MOVIMIENTO Z
# ============================================================

def detectar_z(historial):

    if len(historial) < 15:
        return False

    puntos = list(historial)

    tercio = len(puntos) // 3

    p1 = puntos[0]
    p2 = puntos[tercio]
    p3 = puntos[tercio * 2]
    p4 = puntos[-1]

    # --------------------------------------------
    # La Z aproximadamente hace:
    #
    # ---->
    #    /
    # <----
    #
    # Se observa principalmente el movimiento X.
    # --------------------------------------------

    movimiento1 = p2[0] - p1[0]
    movimiento2 = p3[0] - p2[0]
    movimiento3 = p4[0] - p3[0]

    if (
        abs(movimiento1) > 25
        and abs(movimiento2) > 15
        and abs(movimiento3) > 25
        and movimiento1 * movimiento3 < 0
    ):
        return True

    return False


# ============================================================
# DETECTAR MOVIMIENTO J
# ============================================================

def detectar_j(historial):

    if len(historial) < 12:
        return False

    puntos = list(historial)

    inicio = puntos[0]
    medio = puntos[len(puntos) // 2]
    final = puntos[-1]

    # Movimiento hacia abajo
    baja = medio[1] - inicio[1] > 25

    # Luego curva horizontal
    curva = abs(final[0] - medio[0]) > 20

    if baja and curva:
        return True

    return False


# ============================================================
# CLASIFICAR LETRA
# ============================================================

def reconocer_letra(
        thumb_tip,
        thumb_ip,
        index_tip,
        index_pip,
        middle_tip,
        middle_pip,
        ring_tip,
        ring_pip,
        pinky_tip,
        pinky_pip,
        wrist,
        historial_indice,
        historial_meñique):

    # ------------------------------------------
    # ESTADO DE LOS CUATRO DEDOS PRINCIPALES
    # ------------------------------------------

    indice = dedo_arriba(index_tip, index_pip)
    medio = dedo_arriba(middle_tip, middle_pip)
    anular = dedo_arriba(ring_tip, ring_pip)
    menique = dedo_arriba(pinky_tip, pinky_pip)

    # ------------------------------------------
    # DISTANCIAS
    # ------------------------------------------

    d_thumb_index = distancia(thumb_tip, index_tip)
    d_thumb_middle = distancia(thumb_tip, middle_tip)
    d_thumb_ring = distancia(thumb_tip, ring_tip)
    d_thumb_pinky = distancia(thumb_tip, pinky_tip)

    d_index_middle = distancia(index_tip, middle_tip)
    d_middle_ring = distancia(middle_tip, ring_tip)
    d_ring_pinky = distancia(ring_tip, pinky_tip)

    # ========================================================
    # Z
    # ========================================================

    if indice and not medio and not anular and not menique:
        if detectar_z(historial_indice):
            return "Z"

    # ========================================================
    # J
    # ========================================================

    if menique and not indice and not medio and not anular:
        if detectar_j(historial_meñique):
            return "J"

    # ========================================================
    # A
    # Puño cerrado con pulgar al lado.
    # ========================================================

    if (
        not indice
        and not medio
        and not anular
        and not menique
        and thumb_tip[1] < index_pip[1]
    ):
        return "A"

    # ========================================================
    # B
    # Cuatro dedos arriba y pulgar recogido.
    # ========================================================

    if (
        indice
        and medio
        and anular
        and menique
        and thumb_tip[1] > index_pip[1]
    ):
        return "B"

    # ========================================================
    # C
    # Mano formando una curva / C.
    # ========================================================

    if (
        not indice
        and not medio
        and not anular
        and not menique
        and 60 < d_thumb_index < 180
    ):
        return "C"

    # ========================================================
    # D
    # Índice arriba.
    # Pulgar tocando aproximadamente dedo medio.
    # ========================================================

    if (
        indice
        and not medio
        and not anular
        and not menique
        and d_thumb_middle < 70
    ):
        return "D"

    # ========================================================
    # E
    # Todos flexionados.
    # ========================================================

    if (
        not indice
        and not medio
        and not anular
        and not menique
        and thumb_tip[1] > index_tip[1]
    ):
        return "E"

    # ========================================================
    # F
    # Índice toca pulgar.
    # Medio, anular y meñique extendidos.
    # ========================================================

    if (
        d_thumb_index < 60
        and medio
        and anular
        and menique
    ):
        return "F"

    # ========================================================
    # G
    # Índice extendido horizontal.
    # ========================================================

    if (
        not medio
        and not anular
        and not menique
        and abs(index_tip[1] - thumb_tip[1]) < 70
        and abs(index_tip[0] - thumb_tip[0]) > 40
    ):
        return "G"

    # ========================================================
    # H
    # Índice y medio extendidos juntos horizontalmente.
    # ========================================================

    if (
        indice
        and medio
        and not anular
        and not menique
        and d_index_middle < 80
    ):
        return "H"

    # ========================================================
    # I
    # Solo meñique.
    # ========================================================

    if (
        not indice
        and not medio
        and not anular
        and menique
    ):
        return "I"

    # J está arriba porque requiere movimiento

    # ========================================================
    # K
    # Índice y medio arriba separados.
    # Pulgar entre ellos.
    # ========================================================

    if (
        indice
        and medio
        and not anular
        and not menique
        and d_index_middle > 60
        and d_thumb_middle < 120
    ):
        return "K"

    # ========================================================
    # L
    # Índice vertical + pulgar lateral.
    # ========================================================

    if (
        indice
        and not medio
        and not anular
        and not menique
        and abs(index_tip[0] - thumb_tip[0]) > 80
        and thumb_tip[1] > index_tip[1]
    ):
        return "L"

    # ========================================================
    # M
    # Puño con pulgar debajo de tres dedos.
    # ========================================================

    if (
        not indice
        and not medio
        and not anular
        and not menique
        and d_thumb_ring < d_thumb_index
    ):
        return "M"

    # ========================================================
    # N
    # Puño, pulgar colocado bajo índice/medio.
    # ========================================================

    if (
        not indice
        and not medio
        and not anular
        and not menique
        and d_thumb_middle < d_thumb_ring
    ):
        return "N"

    # ========================================================
    # O
    # Las puntas forman círculo.
    # ========================================================

    if (
        d_thumb_index < 80
        and d_thumb_middle < 100
        and d_thumb_ring < 120
        and d_thumb_pinky < 140
    ):
        return "O"

    # ========================================================
    # P
    # Parecida a K pero apuntando hacia abajo.
    # ========================================================

    if (
        index_tip[1] > index_pip[1]
        and middle_tip[1] > middle_pip[1]
        and not anular
        and not menique
    ):
        return "P"

    # ========================================================
    # Q
    # Índice y pulgar apuntando hacia abajo.
    # ========================================================

    if (
        index_tip[1] > index_pip[1]
        and thumb_tip[1] > thumb_ip[1]
        and not medio
        and not anular
        and not menique
    ):
        return "Q"

    # ========================================================
    # R
    # Índice y medio extendidos y cruzados/juntos.
    # ========================================================

    if (
        indice
        and medio
        and not anular
        and not menique
        and d_index_middle < 40
    ):
        return "R"

    # ========================================================
    # S
    # Puño cerrado con pulgar delante.
    # ========================================================

    if (
        not indice
        and not medio
        and not anular
        and not menique
        and thumb_tip[1] < middle_tip[1]
    ):
        return "S"

    # ========================================================
    # T
    # Puño con pulgar colocado entre índice y medio.
    # ========================================================

    if (
        not indice
        and not medio
        and not anular
        and not menique
        and d_thumb_index < 80
        and d_thumb_middle < 80
    ):
        return "T"

    # ========================================================
    # U
    # Índice y medio arriba y juntos.
    # ========================================================

    if (
        indice
        and medio
        and not anular
        and not menique
        and d_index_middle < 65
    ):
        return "U"

    # ========================================================
    # V
    # Índice y medio arriba separados.
    # ========================================================

    if (
        indice
        and medio
        and not anular
        and not menique
        and d_index_middle > 65
    ):
        return "V"

    # ========================================================
    # W
    # Índice, medio y anular extendidos.
    # ========================================================

    if (
        indice
        and medio
        and anular
        and not menique
    ):
        return "W"

    # ========================================================
    # X
    # Índice doblado en forma de gancho.
    # ========================================================

    if (
        not indice
        and not medio
        and not anular
        and not menique
        and index_tip[1] < index_pip[1] + 80
    ):
        return "X"

    # ========================================================
    # Y
    # Pulgar y meñique abiertos.
    # ========================================================

    if (
        not indice
        and not medio
        and not anular
        and menique
        and d_thumb_pinky > 150
    ):
        return "Y"

    return ""


# ============================================================
# CONFIGURACIÓN MEDIAPIPE
# ============================================================

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Historial para reconocer movimientos
historial_indice = deque(maxlen=20)
historial_meñique = deque(maxlen=20)


# ============================================================
# CÁMARA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


with mp_hands.Hands(

    static_image_mode=False,

    model_complexity=1,

    min_detection_confidence=0.7,

    min_tracking_confidence=0.7,

    max_num_hands=1

) as hands:

    while cap.isOpened():

        success, image = cap.read()

        if not success:
            continue

        # ------------------------------------------
        # EFECTO ESPEJO
        # ------------------------------------------

        image = cv2.flip(image, 1)

        # ------------------------------------------
        # BGR -> RGB
        # ------------------------------------------

        image.flags.writeable = False

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        image.flags.writeable = True


        alto, ancho, _ = image.shape


        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # ============================================
                # DIBUJAR MANO
                # ============================================

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                draw_bounding_box(
                    image,
                    hand_landmarks
                )


                # ============================================
                # PUNTOS MEDIAPIPE
                # ============================================

                wrist = punto(
                    hand_landmarks,
                    0,
                    ancho,
                    alto
                )

                # PULGAR

                thumb_tip = punto(
                    hand_landmarks,
                    4,
                    ancho,
                    alto
                )

                thumb_ip = punto(
                    hand_landmarks,
                    3,
                    ancho,
                    alto
                )

                # ÍNDICE

                index_tip = punto(
                    hand_landmarks,
                    8,
                    ancho,
                    alto
                )

                index_pip = punto(
                    hand_landmarks,
                    6,
                    ancho,
                    alto
                )

                # MEDIO

                middle_tip = punto(
                    hand_landmarks,
                    12,
                    ancho,
                    alto
                )

                middle_pip = punto(
                    hand_landmarks,
                    10,
                    ancho,
                    alto
                )

                # ANULAR

                ring_tip = punto(
                    hand_landmarks,
                    16,
                    ancho,
                    alto
                )

                ring_pip = punto(
                    hand_landmarks,
                    14,
                    ancho,
                    alto
                )

                # MEÑIQUE

                pinky_tip = punto(
                    hand_landmarks,
                    20,
                    ancho,
                    alto
                )

                pinky_pip = punto(
                    hand_landmarks,
                    18,
                    ancho,
                    alto
                )


                # ============================================
                # GUARDAR MOVIMIENTO
                # ============================================

                historial_indice.append(index_tip)
                historial_meñique.append(pinky_tip)


                # ============================================
                # RECONOCER
                # ============================================

                letra = reconocer_letra(

                    thumb_tip,
                    thumb_ip,

                    index_tip,
                    index_pip,

                    middle_tip,
                    middle_pip,

                    ring_tip,
                    ring_pip,

                    pinky_tip,
                    pinky_pip,

                    wrist,

                    historial_indice,
                    historial_meñique
                )


                # ============================================
                # MOSTRAR LETRA
                # ============================================

                if letra != "":

                    cv2.putText(
                        image,
                        letra,
                        (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,
                        (0, 0, 255),
                        8
                    )

                    cv2.putText(
                        image,
                        "Letra detectada",
                        (50, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )


        else:

            # Si no encuentra mano eliminamos historial
            historial_indice.clear()
            historial_meñique.clear()


        # ================================================
        # MOSTRAR CÁMARA
        # ================================================

        cv2.imshow(
            "Reconocimiento Lengua de Senas",
            image
        )


        # ESC para salir
        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()

cv2.destroyAllWindows()