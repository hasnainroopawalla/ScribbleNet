export function predict_doodle(image_string) {
    $.ajax({
        type: 'POST',
        url: '/predict',
        data: {'image_string': image_string},
        success: function (data) {
            alert(data);
            // var div = document.getElementById("op");
            // div.innerHTML = '';
            // a = data.substring(1, data.length - 1).replace(/'/g, '').split(',');
            // txt = a.slice(0, 5); //Objects
            // acc = a.slice(5, 10); //Prediction Accuracy
            // for (var i = 0; i < 5; i++) {
            //     div.innerHTML += txt[i] + " | " + acc[i] + "%" + "<br/>";
            // }
        },
    });
};
