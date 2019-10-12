var orderHistoryData = [{"orderID":'2',"OrderDesc":"Table2","OrderDelDate":"13-9-2017"},{"orderID":'1',"OrderDesc":"Table3","OrderDelDate":"12-9-2019"},{"orderID":'3',"OrderDesc":"Table1","OrderDelDate":"12-8-2019"}]

var listData = [{"ItemName":'phone',"ItemQuality":"1","ItemReqDate":"Sep-13-2019"},{"orderName":'Laptop',"ItemQuality":"1","OrderDelDate":"Sep-22-2019"},{"orderID":'1',"OrderDesc":"Table","OrderDelDate":"Sep-12-2019"}]

var tempData = {"ItemName":'',"ItemQuality":"","ItemReqDate":""};

var orderHistoryTable  = document.getElementById('historyTable');

for(var i in orderHistoryData)
{
	var orderHistoryTbody = document.getElementById('historyBody');
	var orderHistoryTr = document.createElement('tr');
	
	for(var key in orderHistoryData[i])
	{
		var orderHistoryTd = document.createElement('td');
		var txt = document.createTextNode(orderHistoryData[i][key]);
		orderHistoryTd.appendChild(txt)
		orderHistoryTr.appendChild(orderHistoryTd);
	}
	orderHistoryTbody.appendChild(orderHistoryTr);
}

orderHistoryTable.appendChild(orderHistoryTbody);

for(var i in listData)
{
	var listTable  = document.getElementById('listTable');
	var listTr = document.createElement('tr');
	
	for(var key in listData[i])
	{
		var listTd = document.createElement('td');
		var txt = document.createTextNode(listData[i][key]);
		listTd.appendChild(txt)
		listTr.appendChild(listTd);
	}
	var listTd = document.createElement('td');
	listTd.classList.add('removeCell');
	var listspan = document.createElement('span');
	var txt = document.createTextNode("Remove");
	listspan.appendChild(txt);
    listspan.classList.add('removeButton');
    listTd.appendChild(listspan);
	listTr.appendChild(listTd);
	listTable.appendChild(listTr);
}

$('#historyTable tr').on('mouseenter',function(){
	$(this).css({'background-color':'#d9d9d9','color':'white'});
})

$('#historyTable tr').on('mouseleave',function(){
	$(this).css({'background-color':'white','color':'black'});
})


$("#listTable").find("td").attr('contenteditable',"true");

$(document).on('click', '.removeCell', function () {
	$(this).parents('tr').detach();
});

$('#addCell').on('click',function()
{
	var listTable  = document.getElementById('listTable');
	var listTr = document.createElement('tr');
	for(var key in tempData)
	{
		var listTd = document.createElement('td');
		var txt = document.createTextNode("");
		listTd.appendChild(txt)
		listTr.appendChild(listTd);
	}
	var listTd = document.createElement('td');
	listTd.classList.add('removeCell');
	var listspan = document.createElement('span');
	var txt = document.createTextNode("Remove");
	listspan.appendChild(txt);
    listspan.classList.add('removeButton');
    listTd.appendChild(listspan);
	listTr.appendChild(listTd);
	listTable.appendChild(listTr);
	$("#listTable").find("td").attr('contenteditable',"true");
})

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

$('#historyTable .table-up').on('click',function(){
	var value = $(this).parent().attr('value')-1 ;
	var $tbody = $('#historyBody');
	var rows = $('#historyBody tr').get();
	rows.sort(function(a,b){ 
	    var tda = $(a).find('td:eq('+value+')').text(); 
	    var tdb = $(b).find('td:eq('+value+')').text();
	    if(value==2)
	    {
	    	tda = Date.parse(tda);
	    	tdb = Date.parse(tdb);
	    }
	    return tda < tdb ? 1 : tda > tdb ? -1  : 0;           
	});
	  $.each(rows, function(index, row) {
	    $('#historyTable').children('tbody').append(row);
	  });
})

$('#historyTable .table-down').on('click',function(){
	var value = $(this).parent().attr('value')-1 ;
	var $tbody = $('#historyBody');
	var rows = $('#historyBody tr').get();
	rows.sort(function(a,b){ 
	    var tda = $(a).find('td:eq('+value+')').text();
	    var tdb = $(b).find('td:eq('+value+')').text();
	    if(value==2)
	    {
	    	var tda = Date.parse(tda);
	    	var tdb = Date.parse(tdb);
	    }      
	    return tda < tdb ? -1 : tda > tdb ? 1 : 0;           
	});
  $.each(rows, function(index, row) {
    $('#historyTable').children('tbody').append(row);
  });
})


