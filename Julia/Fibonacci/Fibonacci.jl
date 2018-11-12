function fibonacci(n)

	num = fill(0.0, n)

	if n > 0
		num[1] = 1
	end

	if n > 1
		num[2] = 1
	end
	
	if n > 2
		for k = 3:n
			num[k] = num[k - 1] + num[k - 2]
		end
	end
	
	return num

end

n = 10
x = fibonacci(n)

println(x)

