var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

$(window).load(function() {
    $messages.mCustomScrollbar();
    setTimeout(function() {
        //fakeMessage();
    }, 100);
});

function updateScrollbar() {
    $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
        scrollInertia: 10,
        timeout: 0
    });
}

function setDate() {
    d = new Date()
    if (m != d.getMinutes()) {
        m = d.getMinutes();
        $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
    }
}

function insertMessage() {
    msg = $('.message-input').val();
    if ($.trim(msg) == '') {
        return false;
    }
    $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
    setDate();
    $('.message-input').val(null);
    updateScrollbar();
    interact(msg);
    setTimeout(function() {
        //fakeMessage();
    }, 1000 + (Math.random() * 20) * 100);
}

$('.message-submit').click(function() {
    insertMessage();
});

$(window).on('keydown', function(e) {
    if (e.which == 13) {
        insertMessage();
        return false;
    }
})


function interact(message) {
    // loading message
    $('<div class="message loading new"><figure class="avatar"><img src="/static/res/botim.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
    // make a POST request [ajax call]
    // will send the message to the server and expects the json response
    // of the parameter 'type' in the response is text, it will render the text
    // response as normal text,
    // else it will unhide the panel for links and display the links.
    $.post('/message', {
        msg: message,
    }).done(function(reply) {
        if (reply.type == 'text') {
            $(".messages").removeClass("links_on");
            // Message Received
            //  remove loading meassage
            $('.message.loading').remove();
            // Add message to chatbox
            $('<div class="message new"><figure class="avatar"><img src="/static/res/botim.png" /></figure>' + reply['text'] + '</div>').appendTo($('.mCSB_container')).addClass('new');
            setDate();
            updateScrollbar();
        } else {
            $('.message.loading').remove();
            $(".messages").addClass("links_on");
            // Add links to chatbox
            var domString = '<ul>';
            for (var i = 0; i < reply.links.length; i++)
                domString += "<li>" + reply.links[i] + "</li>";
            domString += "</ul>";
            $(".reply_links").html(domString);
            setDate();
            updateScrollbar();
        }

    }).fail(function() {
        alert('error calling function');
    });

}
