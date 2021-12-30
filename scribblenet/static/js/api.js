export function predict_doodle(image_string) {
    $.ajax({
        type: 'POST',
        url: '/predict',
        data: {
            'image_string': image_string
        },
        success: function (data) {
            alert(data);
        },
    });
};
