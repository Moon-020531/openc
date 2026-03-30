import cv2 as cv

img = cv.imread('my_id_card.png')
img_original = img.copy()


drawing = False 
ix, iy = -1, -1
mode = True

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode,img

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            img = img_original.copy()
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
                cv.putText(img, 'FACE', (ix, iy - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0),2)
            cv.putText(img, 'FACE', (ix, iy - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            
img = img_original.copy()
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)


while(1):
    cv.imshow('image', img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):
        cv.imwrite('my_id_card_final.png', img)
    elif k == ord('q'): 
        break
cv.destroyAllWindows()            