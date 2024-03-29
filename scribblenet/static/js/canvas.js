(function () {

    // Get a regular interval for drawing to the screen
    window.requestAnimFrame = (function (callback) {
        return window.requestAnimationFrame ||
            window.webkitRequestAnimationFrame ||
            window.mozRequestAnimationFrame ||
            window.oRequestAnimationFrame ||
            window.msRequestAnimaitonFrame ||
            function (callback) {
                window.setTimeout(callback, 1000 / 60);
            };
    })();

    // Set up the canvas
    var canvas = document.getElementById("sig-canvas");

    var parent = document.getElementById("canvas-parent");
    canvas.width = parent.offsetWidth;
    canvas.height = 400;

    var ctx = canvas.getContext("2d");
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 20;

    // Set up the UI
    var resetBtn = document.getElementById("resetBtn");
    var submitBtn = document.getElementById("predictBtn");

    resetBtn.addEventListener("click", function (e) {
        clearCanvas();
        clearPredictions();
    }, false);

    submitBtn.addEventListener("click", function (e) {
        var dataUrl = canvas.toDataURL("image/png");
        predict_doodle(dataUrl);
    }, false);

    // Set up mouse events for drawing
    var drawing = false;
    var mousePos = {
        x: 0,
        y: 0
    };
    var lastPos = mousePos;
    canvas.addEventListener("mousedown", function (e) {
        drawing = true;
        lastPos = getMousePos(canvas, e);
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        drawing = false;
    }, false);
    canvas.addEventListener("mousemove", function (e) {
        mousePos = getMousePos(canvas, e);
    }, false);

    // Set up touch events for mobile, etc
    canvas.addEventListener("touchstart", function (e) {
        mousePos = getTouchPos(canvas, e);
        var touch = e.touches[0];
        var mouseEvent = new MouseEvent("mousedown", {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }, false);
    canvas.addEventListener("touchend", function (e) {
        var mouseEvent = new MouseEvent("mouseup", {});
        canvas.dispatchEvent(mouseEvent);
    }, false);
    canvas.addEventListener("touchmove", function (e) {
        var touch = e.touches[0];
        var mouseEvent = new MouseEvent("mousemove", {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }, false);

    // Prevent scrolling when touching the canvas
    document.body.addEventListener("touchstart", function (e) {
        if (e.target == canvas) {
            e.preventDefault();
        }
    }, {
        passive: false
    });
    document.body.addEventListener("touchend", function (e) {
        if (e.target == canvas) {
            e.preventDefault();
        }
    }, {
        passive: false
    });
    document.body.addEventListener("touchmove", function (e) {
        if (e.target == canvas) {
            e.preventDefault();
        }
    }, {
        passive: false
    });

    // Get the position of the mouse relative to the canvas
    function getMousePos(canvasDom, mouseEvent) {
        var rect = canvasDom.getBoundingClientRect();
        return {
            x: mouseEvent.clientX - rect.left,
            y: mouseEvent.clientY - rect.top
        };
    }

    // Get the position of a touch relative to the canvas
    function getTouchPos(canvasDom, touchEvent) {
        var rect = canvasDom.getBoundingClientRect();
        return {
            x: touchEvent.touches[0].clientX - rect.left,
            y: touchEvent.touches[0].clientY - rect.top
        };
    }

    // Draw to the canvas
    function renderCanvas() {
        if (drawing) {
            ctx.moveTo(lastPos.x, lastPos.y);
            ctx.lineTo(mousePos.x, mousePos.y);
            ctx.stroke();
            lastPos = mousePos;
        }
    }

    // Reset the canvas
    function clearCanvas() {
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
    }

    // Reset the predictions
    function clearPredictions() {
        var prediction1 = document.getElementById("prediction1");
        var prediction2 = document.getElementById("prediction2");
        var prediction3 = document.getElementById("prediction3");
        prediction1.innerHTML = "*Prediction 1*";
        prediction2.innerHTML = "*Prediction 2*";
        prediction3.innerHTML = "*Prediction 3*";
    }

    // Allow for animation
    (function drawLoop() {
        requestAnimFrame(drawLoop);
        renderCanvas();
    })();

})();

function predict_doodle(image_string) {
    $.ajax({
        type: 'POST',
        url: '/predict',
        data: {
            'image_string': image_string
        },
        success: function (result) {
            display_results(String(result["prediction"]));
        },
    });
};

function display_results(result) {
    var predictions = result.split(",");
    var prediction1 = document.getElementById("prediction1");
    var prediction2 = document.getElementById("prediction2");
    var prediction3 = document.getElementById("prediction3");
    prediction1.innerHTML = predictions[0];
    prediction2.innerHTML = predictions[1];
    prediction3.innerHTML = predictions[2];
}
