var orderHistoryData = [{"orderID":'2',"OrderDesc":"Table2","OrderDelDate":"2008/02/26","price":"500$"},
{"orderID":'1',"OrderDesc":"Table1","OrderDelDate":"2008/01/26","price":"1000$"},
{"orderID":'3',"OrderDesc":"Table3","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'23',"OrderDesc":"groceries","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'4',"OrderDesc":"clothes","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'5',"OrderDesc":"accessories","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'6',"OrderDesc":"Table6","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'7',"OrderDesc":"Table7","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'8',"OrderDesc":"Table8","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'9',"OrderDesc":"Table9","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'10',"OrderDesc":"Table10","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'11',"OrderDesc":"Table11","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'12',"OrderDesc":"Table12","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'13',"OrderDesc":"Table13","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'14',"OrderDesc":"Table14","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'15',"OrderDesc":"Table15","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'16',"OrderDesc":"Table16","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'17',"OrderDesc":"Table17","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'18',"OrderDesc":"Table18","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'19',"OrderDesc":"Table19","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'20',"OrderDesc":"Table20","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'21',"OrderDesc":"Table21","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'22',"OrderDesc":"Table21","OrderDelDate":"2008/09/26","price":"1000$"},
{"orderID":'24',"OrderDesc":"Table24","OrderDelDate":"2013/08/26","price":"1000$"},
{"orderID":'25',"OrderDesc":"Table25","OrderDelDate":"2013/09/26","price":"8000$"}]

var listData = [{"ItemName":'phone',"ItemQuality":"1","ItemReqDate":"Sep-13-2019"},{"orderName":'Laptop',"ItemQuality":"1","OrderDelDate":"Sep-22-2019"},{"orderID":'1',"OrderDesc":"Table","OrderDelDate":"Sep-12-2019"}]

var tempData = {"ItemName":'',"ItemQuality":"","ItemReqDate":""};

var orderHistoryTable  = document.getElementById('historyTable');

function lol()
{
	console.log('lol');
}


$(document).ready(function() {
	//$.fn.dataTable.moment( 'MM-dd-yyyy' );
    $('#historyTable').DataTable();
} );


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

orderHistoryTable.appendChild(orderHistoryTbody,lol);

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

// $('#historyTable .table-up').on('click',function(){
// 	var value = $(this).parent().attr('value')-1 ;
// 	var $tbody = $('#historyBody');
// 	var rows = $('#historyBody tr').get();
// 	rows.sort(function(a,b){ 
// 	    var tda = $(a).find('td:eq('+value+')').text(); 
// 	    var tdb = $(b).find('td:eq('+value+')').text();
// 	    if(value==2)
// 	    {
// 	    	tda = Date.parse(tda);
// 	    	tdb = Date.parse(tdb);
// 	    }
// 	    return tda < tdb ? 1 : tda > tdb ? -1  : 0;           
// 	});
// 	  $.each(rows, function(index, row) {
// 	    $('#historyTable').children('tbody').append(row);
// 	  });
// })

// $('#historyTable .table-down').on('click',function(){
// 	var value = $(this).parent().attr('value')-1 ;
// 	var $tbody = $('#historyBody');
// 	var rows = $('#historyBody tr').get();
// 	rows.sort(function(a,b){ 
// 	    var tda = $(a).find('td:eq('+value+')').text();
// 	    var tdb = $(b).find('td:eq('+value+')').text();
// 	    if(value==2)
// 	    {
// 	    	var tda = Date.parse(tda);
// 	    	var tdb = Date.parse(tdb);
// 	    }      
// 	    return tda < tdb ? -1 : tda > tdb ? 1 : 0;           
// 	});
//   $.each(rows, function(index, row) {
//     $('#historyTable').children('tbody').append(row);
//   });
// })


