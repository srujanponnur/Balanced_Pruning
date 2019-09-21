$('.firstSection,.secondSection,.thirdSection,.fourthSection').click(function()
{
   if($(this).find('.panel').is(":hidden"))
   {
   	$(this).find('.section-title-second').hide();
   	$(this).find('.panel').slideDown('slow');
   }
   else
   {
   	$(this).find('.section-title-second').show();
   	$(this).find('.panel').slideUp('fast');
   }
   
});

$('.fifthSection').dblclick(function()
{
   if($(this).find('.panel').is(":hidden"))
   {
   	$(this).find('.section-title-second').hide();
   }
   else
   {
   	$(this).find('.section-title-second').show();
   }
   $(this).find('.panel').slideToggle('slow');
   
});


$('.form-button').mouseleave(function()
{
   $('.sixthSection').find('.section-title-second').show();
   $('.sixthSection').find('.panel').slideUp('slow',function(){alert('Cannot Submit :p')});

});


$('.form-button').mouseleave(function()
{
  $('.sixthSection').find('.section-title-second').hide();
  $('.sixthSection').find('.panel').slideDown('slow');
});

$('input').keypress(function(){
	$(this).prev().slideUp(2000).slideDown(2000);
})

$('input').keydown(function(){
	$('.animLabel').animate({'font-size':'5px','font-weight':'200'},2000,function(){
		$(this).css({'font-size':'15px','font-weight':'400'});
	})
})

$('form').submit(function(){

	alert('Done!');
})