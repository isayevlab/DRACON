    (function() {
          var fn = function() {
            Bokeh.safely(function() {
              (function(root) {
                function embed_document(root) {
                  
                var render_items = [{"docid":"2aadd1a8-0158-4c57-8a60-c05a31f0157f","root_ids":["3030"],"roots":{"3030":"c7afbef7-8715-4574-b1aa-3b2c19c0c731"}}];
                root.Bokeh.embed.embed_items(docs_json, render_items);
              
                }
                if (root.Bokeh !== undefined) {
                  embed_document(root);
                } else {
                  var attempts = 0;
                  var timer = setInterval(function(root) {
                    if (root.Bokeh !== undefined) {
                      clearInterval(timer);
                      embed_document(root);
                    } else {
                      attempts++;
                      if (attempts > 100) {
                        clearInterval(timer);
                        console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                      }
                    }
                  }, 10, root)
                }
              })(window);
            });
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();