$('.newbtn1').bind("click" , function () {
    $('#pic1').click();
});

$('.newbtn2').bind("click" , function () {
    $('#pic2').click();
});

function readURL1(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#blah1').attr('src', e.target.result);
        };

        reader.readAsDataURL(input.files[0]);
    }
}

function readURL2(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#blah2').attr('src', e.target.result);
        };

        reader.readAsDataURL(input.files[0]);
    }
}
