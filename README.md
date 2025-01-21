# wine_sales_optimization
Price and sales optimization of wine for Art Noir events.

## Problem Statement

The following repository processes simulated data that represents the buying behavior of wine by attendees at the recent 3
events Art Noir hosted. The goal of the problem is to first estimate how much wine to purchase based on the past behaviors and the 
expected number of attendees that will attend the next event. Once this problem is solved, the estimated expense for the
wine can be calculated. The next problem to solve, is what to sell each glass of wine that maximizes profit while also not curving
demand.

## Data

### Wime

This table is a dimension table that describes what wine has been sold at all three past events

- wine_id: int -> the primary key of the wine table data
- brand: str -> Name of the wine
- size-ml: int the amount of wine that brand's bottle holds. in mililiters
- type: str -> vals will either be white, red, or sparkling
- cost: float -> the cost of that brand's bottle

### Drink_counts

This table is an aggreagetd table on how much wine each attendee bought at each event

- cust_id: int -> customer id
- First Name: str -> customer's first name
- Last Name: str -> customer's last name
- Drink Count: int -> The number of wine ordered at that event
- event number: int -> event id

### Orders

- order_date: datetime -> The time the order transaction occurred
- wine_id: int -> the wine id
- event number -> the event number
- order_id: int -> order id
- cust_id: customer id


