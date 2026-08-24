
function btnSetActive(btn, btn_group) {
  this.className += " active";
  var btns = btn_group.getElementsByClassName("btn");
  for (var i = 0; i < btns.length; i++) {
    w3RemoveClass(btns[i],"btn_active");
  }
  w3AddClass(btn,"btn_active");
}

function sortDirection(sort_class="cat_id", sort_direction="acc") {
  var container = document.getElementById("topic_placeholder");
  var elements = container.childNodes;

  var elements_arr = Array.prototype.slice.call(elements, 0);

  elements_arr.sort(function(a, b) {
      var a_ord = a.getElementsByClassName(sort_class)[0].innerHTML;
      var b_ord = b.getElementsByClassName(sort_class)[0].innerHTML;

      if (sort_class == "score") {
        if (sort_direction == "acc") {
          if (isNaN(a_ord)) { a_ord = 0; }
          if (isNaN(b_ord)) { b_ord = 0; }  
          return b_ord - a_ord;
        } else { 
          if (isNaN(a_ord)) { a_ord = 100000000; }
          if (isNaN(b_ord)) { b_ord = 100000000; }  
          return a_ord - b_ord;
        }
      }

      if (sort_class == "cat_id") {
        if (a_ord > b_ord) return 1;
        if (a_ord < b_ord) return -1;
        return 0;
      }

  });

  // Change the order
  container.innerHTML = "";
  for(var i = 0, l = elements_arr.length; i < l; i++) {
      container.appendChild(elements_arr[i]);
  }

}


filterSelection("all")
function filterSelection(c) {
  var filtered_div, i;
  filtered_div = document.getElementsByClassName("filter_topic");

  if (c == "all") {
    for (i = 0; i < filtered_div.length; i++) {
      w3RemoveClass(filtered_div[i], "hidden");
    }
  }
  else {
    // Add the "show" class (display:block) to the filtered elements, and remove the "show" class from the elements that are not selected
    for (i = 0; i < filtered_div.length; i++) {
      w3RemoveClass(filtered_div[i], "hidden");
      var cat_id = filtered_div[i].getElementsByClassName("cat_id")[0].innerHTML 
      if (cat_id[0] != c) w3AddClass(filtered_div[i], "hidden");
    }
  }
}

// Show filtered elements
function w3AddClass(element, name) {
  var i, arr1, arr2;
  arr1 = element.className.split(" ");
  arr2 = name.split(" ");
  for (i = 0; i < arr2.length; i++) {
    if (arr1.indexOf(arr2[i]) == -1) {
      element.className += " " + arr2[i];
    }
  }
}

// Hide elements that are not selected
function w3RemoveClass(element, name) {
  var i, arr1, arr2;
  arr1 = element.className.split(" ");
  arr2 = name.split(" ");
  for (i = 0; i < arr2.length; i++) {
    while (arr1.indexOf(arr2[i]) > -1) {
      arr1.splice(arr1.indexOf(arr2[i]), 1);
    }
  }
  element.className = arr1.join(" ");
}