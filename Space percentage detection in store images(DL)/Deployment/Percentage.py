def Calculate_Space(results):
    result=results[0]
    boxes = result.boxes

    class_0_annotations = []
    class_1_annotations = []

    for i in range(len(boxes.cls)):
        xmin, ymin, xmax, ymax = boxes.xyxy[i]
        class_label = int(boxes.cls[i])

        annotation = [xmin.item(), ymin.item(), xmax.item(), ymax.item()]

        if class_label == 0:
            class_0_annotations.append(annotation)
        elif class_label == 1:
            class_1_annotations.append(annotation)

    total_area_class_0 = sum((anno[2] - anno[0]) * (anno[3] - anno[1]) for anno in class_0_annotations)
    total_area_class_1 = sum((anno[2] - anno[0]) * (anno[3] - anno[1]) for anno in class_1_annotations)

    class_1_ratio = total_area_class_1 / (total_area_class_0 + total_area_class_1)
    class_1_ratio_percentage = class_1_ratio * 100

    return class_1_ratio_percentage