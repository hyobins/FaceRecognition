#Define Function

#Load model which is alreay learned
def load_model(pb_path, image_size=(160,160)):
    tf.reset_default_graph()

    single_image = tf.placeholder(tf.int32, (None,None,3))
    float_image = tf.cast(single_image, tf.float32)
    float_image = float_image/255
    batch_image = tf.expand_dims(flost_image, 0)
    resized_image = tf.image.resize(batch_image, image_size)

    phase_train = tf.placeholder_with_default(False, shape=[])

    input_map = {'image_batch':resized_image, 'phase_train':phase_train}
    model = facenet.load_model(pb_path, input_map)

    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

    return single_image, embeddings

#get the path of vedio, return image

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

#calculate btw two vectors
def calc_distance(embedding1, embedding2):
    diff = np.subtract(embedding1, embedding2)
    dist = np.sum(np.square(diff),0)

    return dist

#Load image getting path as arg
def MyImread(path):
    bar_img = cv2.image(path)
    assert bar_img is not None, "Fail to load vedio"

    rgb_img = cv2.cvtColor(bar_img, cv2.COLOR_BGR2RGB)
    return rgb_img

#crop faces to bounding boxes. Return image and location
def crop_faces(image, pnet, rnet, onet):
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    margin = 44
    image_size = 160

    h,w,_ = np.shape(image)

    bounding_boxes, points = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

    faces = []
    for box in bounding_boxes:
        box = np.int32(box)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(box[0]-margin/2,0)
        bb[1] = np.maximum(box[1]-margin/2,0)
        bb[2] = np.maximum(box[2]-margin/2,w)
        bb[3] = np.maximum(box[3]-margin/2,h)
        cropped = image[bb[1]:bb[3],bb[0]:bb[2],:]
        scaled = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        faces.append(scaled)

    return faces, bounding_boxes

#calculate sample's embedding
single_image, embeddings = load_model("./models/20180402-114759.pb")
sess = tf.Session()

path_me = glob.glob("./data/faces/hyobin/*")
embed_me = []

for path in path_me:
    img = load_image(path)
    result = sess.run(embeddings, feed_dict={single_image:img})
    result = result[0]
    embed_me.append(result)

embed_me = np.array(embed_me)

#load vedio
cap = cv2.VideoCapture('./data/faces/juun.mov')

#load model and set session
tf.reset_default_graph()
single_image, embeddings = load_image("./models/20180402-114759.pb")
sess = tf.Session()
pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

while(cap.isOpened()):
    ret, frame = cap.read()

    if(ret == True):
        """Adjusting vedio size"""
        frame = cv2.resize(frame, (400,225))

        """Getting faces"""
        faces, bounding_boxes = crop_faces(frame, pnet, rnet, onet)

        """Getting vector of sample face and calculate distance"""
        for i in range(len(faces)):
            result = sess.run(embeddings, feed_dict={single_image:faces[i]})
            result = result[0]
            distance_th = 1.2
            distance1 = calc_distance(embed_me[0], result)
            distance2 = calc_distance(embed_me[1], result)
            distance3 = calc_distance(embed_me[2], result)
            distance4 = calc_distance(embed_me[3], result)
            distance5 = calc_distance(embed_me[4], result)
            avg_distance = (distance1+distance2+distance3+distance4+distance5)/5
            if(avg_distance < distance_th):
                print("It's me")
                box = bounding_boxes[i]
                box = np.int32(box)
                p1 = (box[0], box[1])
                p2 = (box[2], box[3])
                cv2.rectangle(frame, p1, p2, color=(0,255,0))
            else:
                print("It's not me")
                box = bounding_boxes[i]
                box = np.int32(box)
                p1 = (box[0], box[1])
                p2 = (box[2], box[3])
                cv2.rectangle(frame, p1, p2, color=(0,0,255))

            cv2.imshow('detected', frame)
        else:
            break
        if cv2.waitKey(1) & OxFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindow()
