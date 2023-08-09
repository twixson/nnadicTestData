#' Combine the eight different training datasets into one
#'
#' The datasets were split for faster loading but are useful combined
#'
#' @return puts `train_data_four` into the environment
#' @export
#'
#' @examples
#' combine_training_data()
combine_training_data <- function(){
  train_data_four <<- array(NA, dim = c(32000, 500, 2))

  for(i in 1:8){
    begin_index <- (i-1) * 4000 + 1
    end_index <- i*4000
    train_data_four[begin_index:end_index, , ] <<-
      get(paste0("train_data_four", i))
  }
}
