test:
	python -m pytest -s tests/

coverage:
	coverage run --source=instafilter -m pytest tests/
	coverage report -m
	coverage html
	xdg-open htmlcov/index.html

lint:
	black instafilter tests --line-length 80

clean:
	rm -rvf cover
